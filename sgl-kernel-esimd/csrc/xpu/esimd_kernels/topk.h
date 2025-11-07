#include "utils.h"


template<uint32_t TD>
inline void esimd_topk_local(
    uint8_t* input,
    uint8_t* input_idx,
    // uint8_t* debug_buf,
    // uint8_t* debug_buf1,
    // uint8_t* debug_buf2,
    uint8_t* out_ordered,
    uint8_t* out_idx,
    uint32_t RCOUNT,
    sycl::queue& dpcpp_queue) {

    // assmue RCOUNT*total_count = RCOUNT * TD * 64, if TD=128, RCOUNT*total_count = RCOUNT*8192
    // assume BARREL_CNT is 16
    // assume topk is 2048
    constexpr int topk = 2048;
    constexpr int total_count = TD * 64;
    int n_threads = 64;

    sycl::range<1> GlobalRange(RCOUNT * n_threads);
    sycl::range<1> LocalRange(n_threads);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    // input and input_idx size is (RCOUNT, total_count)  fp16, uint32_t
    // out_ordered and out_idx size is (RCOUNT, topk)   fp16, uint32_t

    // in SLM:
    // 
    // barrel_buf  size is (64 * 16) = (1024)
    // more right more new data.  BARREL_CNT is barrel num, 0,1,2,...,15
    // barrel_reduce_buf size is (64)
    //
    // input and input_idx (total_count)

    dpcpp_queue.submit([&](handler& cgh) {
    cgh.parallel_for(
      Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {

        constexpr int barrel_buf_size = 64 * 16 * sizeof(uint32_t);
        constexpr int barrel_reduce_buf_size = 64 * sizeof(uint32_t);
        constexpr int input_size = total_count * sizeof(fp16);
        constexpr int input_idx_size = total_count * sizeof(uint32_t);
        __ESIMD_NS::slm_init(barrel_buf_size + barrel_reduce_buf_size + input_size + input_idx_size);

        constexpr int barrel_buf_offset = 0;
        constexpr int barrel_reduce_buf_offset = barrel_buf_offset + barrel_buf_size;
        constexpr int input_offset = barrel_reduce_buf_offset + barrel_reduce_buf_size;
        constexpr int input_idx_offset = input_offset + input_size;

        int gid = ndi.get_group(0);
        int tid = ndi.get_local_id(0);

        // get input and input idx
        int cur_input_offset = input_offset;
        int cur_input_idx_offset = input_idx_offset;
        int cur_output_offset = 0;
        int cur_output_idx_offset = 0;

        simd<fp16, TD> input_cur = block_load<fp16, TD>((fp16*)input + gid * total_count + TD * tid);
        simd<uint32_t, TD> input_idx_cur = block_load<uint32_t, TD>((uint32_t*)input_idx + gid * total_count + TD * tid);

        // slm_block_store<fp16, TD>(cur_input_offset + TD * tid * sizeof(fp16), input_cur);
        // slm_block_store<uint32_t, TD>(cur_input_idx_offset + TD * tid * sizeof(uint32_t), input_idx_cur);
        constexpr int step_count = 4;

        for (int step_idx = 0; step_idx < step_count; step_idx++)
        {
          cur_input_offset = input_offset;
          cur_input_idx_offset = input_idx_offset;
          cur_output_offset = input_offset;
          cur_output_idx_offset = input_idx_offset;
          // select field due to steps
          uint16_t field = (0xf << (step_idx*4));

          if (step_idx > 0)
          {
            input_cur = slm_block_load<fp16, TD>(cur_input_offset + TD * tid * sizeof(fp16));
            input_idx_cur = slm_block_load<uint32_t, TD>(cur_input_idx_offset + TD * tid * sizeof(uint32_t));
          }
          
          // 4 substeps radix_index, cumsum, cumsum_2, scan
          // radix_index ------------------------------------------------------------------------------------------------
          simd<uint16_t, TD> input_barrel_indexing;
          input_barrel_indexing = input_cur.template bit_cast_view<uint16_t>().template select<TD, 1>(0) & field;
          input_barrel_indexing = (input_barrel_indexing >> (step_idx*4));

          simd<uint32_t, 16> local_barrel{0};

          #pragma unroll
          for (int k = 0; k < TD; k++)
          {
            local_barrel[input_barrel_indexing[k]] = local_barrel[input_barrel_indexing[k]] + 1;
          }

          slm_block_store<uint32_t, 16>(barrel_buf_offset + tid * 16 * sizeof(uint32_t), local_barrel);
          
          //block_store<uint32_t, 16>((uint32_t*)debug_buf + tid * 16, local_barrel);
          
          barrier();
          // cumsum -----------------------------------------------------------------------------------------------------
          simd<uint32_t, 16> offsets(0, 16*sizeof(uint32_t));
          int barrel_id = tid / 4;
          int seq_sub_id = tid % 4;
          offsets = offsets + barrel_buf_offset + (barrel_id + 16 * 16 * seq_sub_id)* sizeof(uint32_t);
          local_barrel = slm_gather<uint32_t, 16>(offsets);
          
          #pragma unroll
          for (int k = 1; k < 16 ; k++)
          {
            local_barrel[k] = local_barrel[k] + local_barrel[k-1];
          }
          slm_block_store<uint32_t, 1>(barrel_reduce_buf_offset + tid * sizeof(uint32_t) , local_barrel.select<1, 1>(15));

          //block_store<uint32_t, 1>((uint32_t*)debug_buf1 + tid, local_barrel.select<1, 1>(15));

          barrier();
          // cumsum_2 ---------------------------------------------------------------------------------------------------

          simd<uint32_t, 64> local_reduce_barrel = slm_block_load<uint32_t, 64>(barrel_reduce_buf_offset);

          simd<uint32_t, 64> result_vec;
          // 1. 创建一个从 0 到 31 的序列
          simd<uint32_t, 64> sequence = simd<uint32_t, 64>(0, 1);
          // 2. 根据变量 N 生成掩码
          simd_mask<64> mask = sequence < tid;
          // 3. 使用 where 进行条件赋值
          result_vec.merge(0xffffffff, 0, mask);
          local_reduce_barrel = local_reduce_barrel & result_vec;
          uint32_t reduced_barrel_sum = reduce<uint32_t, uint32_t, 64, std::plus<>>(local_reduce_barrel, std::plus<>());

          local_barrel = local_barrel + reduced_barrel_sum;

          slm_scatter<uint32_t, 16>(offsets, local_barrel);

          barrier();
          // scan -------------------------------------------------------------------------------------------------------

          local_barrel = slm_block_load<uint32_t, 16>(barrel_buf_offset + tid * 16 * sizeof(uint32_t));

          //block_store<uint32_t, 16>((uint32_t*)debug_buf2 + tid * 16, local_barrel);

          #pragma unroll
          for (int k = TD-1; k >= 0 ; k--)
          {
            fp16 out_ordered_cur = input_cur[k];
            uint32_t out_idx_cur = input_idx_cur[k];
            
            uint32_t new_offset = local_barrel[input_barrel_indexing[k]];
            local_barrel[input_barrel_indexing[k]] = new_offset - 1;

            slm_block_store<fp16, 1>(cur_output_offset + (new_offset - 1) * sizeof(fp16), out_ordered_cur);
            slm_block_store<uint32_t, 1>(cur_output_idx_offset + (new_offset - 1) * sizeof(uint32_t), out_idx_cur);
          }

          barrier();
        } // step loop

        // final output
        int final_output_offset = cur_output_offset + (total_count - topk) * sizeof(fp16);
        int final_output_idx_offset = cur_output_idx_offset + (total_count - topk) * sizeof(uint32_t);

        // 2048/64=32 per thread write out:
        simd<fp16, 32> output_cur = slm_block_load<fp16, 32>(final_output_offset + tid * 32 * sizeof(fp16));
        simd<uint32_t, 32> output_idx_cur = slm_block_load<uint32_t, 32>(final_output_idx_offset + tid * 32 * sizeof(uint32_t));
        
        block_store<fp16, 32>((fp16*)out_ordered + gid * topk + tid * 32, output_cur);
        block_store<uint32_t, 32>((uint32_t*)out_idx + gid * topk + tid * 32, output_idx_cur);
      });
  });

}
