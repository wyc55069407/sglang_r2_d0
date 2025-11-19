#include "utils.h"

// ref impl
#define FP8_E4M3_TO_FLOAT(in) \
  h = (uint16_t)in << 8; \
  sign = h & 0x8000; \
  h = h & 0x7FFF; \
  h |= 0x0080; \
  h = h >> 1; \
  result.u = sign | h; \
  v = static_cast<scalar_t>(result.f); \

#define MAX_HD    1024
#define MAX_PPG   1024
#define MAX_T   64

#define WE  4
#define WM  3

//A (M, K),  W (N, K)
template<typename IT, uint32_t NT, uint32_t HD, uint32_t PPG, uint32_t MAX_INPUT_M>
void GEMV_a16_wfp16_block(
  uint8_t* input_data,
  uint8_t* weight_data, 
  uint8_t* bias_data,
  uint8_t* q_scale_data,
  uint8_t* output_data,
  uint32_t M,
  uint32_t N,
  uint32_t K,
  uint32_t batch,
  uint32_t has_bias,
  float softmax_scale,
  sycl::queue& q) {

    assert(K % HD == 0);
    assert(K % 128 == 0);
    static_assert(HD <= MAX_HD);
    static_assert(PPG <= MAX_PPG);
    static_assert(NT <= MAX_T);
    assert(M <= MAX_INPUT_M);
    static_assert(sizeof(IT) == sizeof(fp16));

    constexpr uint32_t CHUNK = NT * HD;
    uint32_t chunk_n = (K + CHUNK - 1) / CHUNK;
    uint32_t active_thread_num_last_chunk = NT;
    constexpr uint32_t reduce_result_n_per_t = (MAX_INPUT_M + NT - 1) / NT; // per thread reduce num
    constexpr uint32_t reduce_result_t_n = MAX_INPUT_M / reduce_result_n_per_t; // thread num used for reduce
    if (K % CHUNK != 0)
    {
      active_thread_num_last_chunk = (K % CHUNK + HD - 1) / HD;
    }
    uint32_t last_thread_id = active_thread_num_last_chunk - 1;
    uint32_t wg_n = (N + PPG -1) / PPG;

    sycl::range<2> GlobalRange(wg_n * NT, batch); // N/ppg, batch
    sycl::range<2> LocalRange(NT, 1);   // NT threads
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    uint32_t last_wg_id = wg_n - 1;

    sycl::event e = q.submit([&](handler& cgh) {
      cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{

      // SLM layout  IT
      // (M, NT, ppg)
      __ESIMD_NS::slm_init(MAX_INPUT_M * PPG * NT * sizeof(IT));

      int hh = ndi.get_local_id(0);
      int h = ndi.get_group(0);
      int b = ndi.get_group(1);

      const IT *  input_ptr = ((IT*)input_data) + hh * HD + b * M * K;
      const fp16 * weight_ptr = ((fp16 *)weight_data) + h * PPG * K + hh * HD + b * N * K;
      const uint32_t slmAccumulationOffset = hh * PPG * sizeof(IT);

      simd<IT, PPG*MAX_INPUT_M> slmAccumulationTemp;  // slmAccumulationTemp shape (PPG)
      simd<IT, HD*MAX_INPUT_M> input;

      slmAccumulationTemp = 0;
      // Loop CHUNK
      for (int ck = 0; ck < chunk_n; ck++) {
        if (ck < chunk_n - 1 || hh < active_thread_num_last_chunk)  // only "not last chunk" or "lask chunk active threads" need execution
        {
          if (MAX_INPUT_M == 1)
          {
            input.template select<HD, 1>(0) = block_load<IT, HD>(input_ptr + CHUNK * ck);
          }
          else
          {
            // load input    (HD)
            for (int ii = 0; ii < M; ii++) {
              input.template select<HD, 1>(HD * ii) = block_load<IT, HD>(input_ptr + CHUNK * ck + ii * K);
            }
          }

          // Loop PPG
          #pragma unroll
          for (int pp = 0; pp < PPG; pp++) {
            if (h < last_wg_id || (pp + PPG*h < N))  // only "not last wg" or "last wg but ppg in range" need execution
            {
                // read W   (HD)
                simd<fp16, HD> weight = block_load<fp16, HD>(weight_ptr + pp * K + CHUNK * ck);

                if (MAX_INPUT_M == 1)
                {
                    slmAccumulationTemp[pp] += sycl::ext::intel::esimd::detail::sum<IT, IT, HD>(weight * input.template select<HD, 1>(0));
                }
                else
                {
                    for (int ii = 0; ii < M; ii++) {
                        slmAccumulationTemp[pp + ii * PPG] += sycl::ext::intel::esimd::detail::sum<IT, IT, HD>(weight * input.template select<HD, 1>(HD * ii));
                    }
                }
            } // only "not last wg" or "last wg but ppg in range" need execution
          } // Loop PPG
        } // only "not last chunk" or "lask chunk active threads" need execution
      } // Loop CHUNK

        // write to SLM
        if (MAX_INPUT_M == 1)
        {
          slm_block_store<IT, PPG>(slmAccumulationOffset, slmAccumulationTemp.template select<PPG, 1>(0));
        }
        else
        {
          for (int ii = 0; ii < M; ii++) {
            slm_block_store<IT, PPG>(slmAccumulationOffset + ii * PPG * NT * sizeof(IT), slmAccumulationTemp.template select<PPG, 1>(PPG * ii));
          }
        }

        barrier();

        // reduce (M, NT, PPG) results to(M, 1, PPG)
        if (hh < reduce_result_t_n)
        {
          #pragma unroll
          for (int i = 0; i < reduce_result_n_per_t; i++) {
            uint32_t ii = hh * reduce_result_n_per_t + i;
            if (ii < M)
            {
              simd<IT, PPG> final_result;  // shape (PPG)
              simd<IT, PPG> bias; // shape (PPG)

              final_result = 0;

              // shape (NT * PPG)
              simd<IT, NT * PPG> result_to_reduce = slm_block_load<IT, NT * PPG>(ii * PPG * NT * sizeof(IT));
              #pragma unroll
              for (int pp = 0; pp < PPG; pp++) {
                final_result[pp] = sycl::ext::intel::esimd::detail::sum<IT, IT, NT>(result_to_reduce.template select<NT, PPG>(pp));
              }
              
              if (q_scale_data)
              {
                simd<IT, PPG> q_scale_cur = block_load<IT, PPG>(((IT*)q_scale_data) + h * PPG + b * M * N + ii * N);
                final_result = final_result * q_scale_cur;
                final_result = final_result * softmax_scale;
              }

              // load bias  (PPG)
              if (has_bias)
              {
                bias.template select<PPG, 1>(0) = block_load<IT, PPG>(((IT*)bias_data) + h * PPG + b * N);
                // writeOut  (PPG)
                block_store<IT, PPG>(((IT*)output_data) + h * PPG + b * M * N + ii * N, final_result + bias);
              }
              else
              {
                // writeOut  (PPG)
                block_store<IT, PPG>(((IT*)output_data) + h * PPG + b * M * N + ii * N, final_result);
              }
            } // ii < M
          } // loop reduce_result_n_per_t
        } // check reduce_result_t_n

    });
  });
}