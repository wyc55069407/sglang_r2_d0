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
template<typename IT, uint32_t NT, uint32_t HD, uint32_t PPG, typename ITS, 
uint32_t scale_block_size_N, uint32_t scale_block_size_K, bool DEQUANT_TEST, uint32_t MAX_INPUT_M>
inline void GEMV_a16_wfp8_block(
  uint8_t* input_data,
  uint8_t* weight_data, 
  uint8_t* weight_scale_data,
  uint8_t* bias_data,
  uint8_t* output_data,
  uint32_t M,
  uint32_t N,
  uint32_t K,
  uint32_t batch,
  uint32_t has_bias,
  sycl::queue& q) {
    // Limitation:   K % 128 == 0   M <= MAX_INPUT_M    K % scale_block_size_K == 0 or K
    // NT <= 64    IT is fp16    ITS is fp16   K % HD == 0    HD <= 1024    PPG <= 1024     all hyper params must be 2^
    // batch here is for batch GEMV in absorb. indicated as head.
    assert(K % HD == 0);
    assert(K % 128 == 0);
    static_assert(HD <= MAX_HD);
    static_assert(PPG <= MAX_PPG);
    static_assert(NT <= MAX_T);
    assert(M <= MAX_INPUT_M);
    static_assert(sizeof(IT) == sizeof(fp16));
    static_assert(sizeof(ITS) == sizeof(fp16));

    constexpr uint32_t CHUNK = NT * HD;
    constexpr uint32_t scale_k_n_pert = (HD + scale_block_size_K - 1) / scale_block_size_K;
    uint32_t chunk_n = (K + CHUNK - 1) / CHUNK;
    uint32_t scale_stride = (K + scale_block_size_K - 1) / scale_block_size_K;
    uint32_t scale_stride_n = (N + scale_block_size_N - 1) / scale_block_size_N;
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
      const uint8_t * weight_ptr = ((uint8_t *)weight_data) + h * PPG * K + hh * HD + b * N * K;
      const ITS *  weight_scale_ptr = ((ITS*)weight_scale_data) + hh * HD / scale_block_size_K + b * scale_stride_n * scale_stride;
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
                simd<uint8_t, HD> weight_quanted = block_load<uint8_t, HD>(weight_ptr + pp * K + CHUNK * ck);
                // read W_scale   (HD / scale_block_size_K)
                simd<ITS, scale_k_n_pert> weight_scale = 
                  block_load<ITS, scale_k_n_pert>(weight_scale_ptr + (h *PPG + pp) / scale_block_size_N * scale_stride + CHUNK * ck / scale_block_size_K);

                simd<IT, HD> weight{0};
                // FP8_E4M3_TO_FLOAT esimd impl
                {
                  simd<uint16_t, HD> result{0.0};
                  {
                    simd<uint8_t, HD> x = weight_quanted;
                    constexpr uint16_t weo = 5;
                    constexpr uint16_t wmo = 10;
                    // constexpr uint16_t ifNaN = 0x7F800001;
                  
                    auto is_zero = (x == 0);
                    // auto is_nan = (x == 0x80);
                  
                    simd<uint16_t, HD> mantissa = x & ((1 << WM) - 1);
                    simd<uint16_t, HD> exponent = (x & 0x7F) >> WM;

                    auto zero_exponent = (exponent == 0);
                    simd<uint16_t, HD> mantissa_subnormal = mantissa;
                    simd<uint16_t, HD> exponent_subnormal = exponent;
                    // subnormal input
                    {
                      // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above

                      simd<uint16_t, HD> renorm_shift_vec;
                      simd<uint16_t, HD> vec = mantissa;
                      {
                        simd<float, HD> vec_float = vec;

                        simd<uint32_t, HD> vec_uint = vec_float.template bit_cast_view<uint32_t>();
                        simd<uint32_t, HD> exponent_tmp = (vec_uint >> 23) & 0xFF;

                        simd<uint32_t, HD> lz = 158 - exponent_tmp; // 158 = 127 + 31

                        renorm_shift_vec = lz;
                      }
                      simd<uint16_t, HD> renorm_shift = renorm_shift_vec;

                      // renorm_shift111 = (IT)renorm_shift;

                      simd<uint16_t, HD> sh = 1 + renorm_shift - (32 - WM);
                      mantissa_subnormal <<= sh;
                      exponent_subnormal += 1 - sh;
                      mantissa_subnormal &= ((1 << WM) - 1);
                    }

                    mantissa.merge(mantissa_subnormal, zero_exponent);
                    exponent.merge(exponent_subnormal, zero_exponent);
                  
                    const uint16_t exp_low_cutoff = (1 << (weo - 1)) - (1 << (WE - 1));
                    exponent += exp_low_cutoff;
                    mantissa <<= wmo - WM;
                  
                    simd<uint16_t, HD> sign = x >> 7;
                    simd<uint16_t, HD> retval = (sign << 15) | (exponent << 10) | mantissa;

                    retval.merge(0, is_zero);
                    // retval.merge(ifNaN, is_nan);
              
                    result = retval;
                  }
              
                  weight = result.template bit_cast_view<IT>();
                }

                // dequant
                #pragma unroll
                for (int dq = 0; dq < scale_k_n_pert; dq++) {
                  weight.template select<HD/scale_k_n_pert, 1>(HD/scale_k_n_pert * dq) = weight.template select<HD/scale_k_n_pert, 1>(HD/scale_k_n_pert * dq) * (IT)weight_scale[dq];
                }

                if (DEQUANT_TEST)
                {
                  block_store<IT, HD>(((IT*)output_data) + pp * K + CHUNK * ck + h * PPG * K + hh * HD + b * N * K, weight);
                  // ((IT*)output_data)[0] = renorm_shift111;
                }
                else
                {
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
                }
            } // only "not last wg" or "last wg but ppg in range" need execution
          } // Loop PPG
        } // only "not last chunk" or "lask chunk active threads" need execution
      } // Loop CHUNK

      if (!DEQUANT_TEST)
      {
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
      } // !DEQUANT_TEST

    });
  });
}

// mat mul
// simd<IT, 16> cc = 0.0f; 
// #pragma unroll
// for (int k = 1; k < (HD/16); k++) {
//   cc.select<16, 1>(0) += weight.select<16, 1>(16*k) * input.select<16, 1>(16*k);
// }
// cc.select<8, 1>(0) += cc.select<8, 1>(8);
// cc.select<4, 1>(0) += cc.select<4, 1>(4);
// cc.select<2, 1>(0) += cc.select<2, 1>(2);
// slmAccumulationTemp[pp] += cc[0] + cc[1];


//A (M, K),  W (N, K)
template<typename IT, uint32_t NT, uint32_t HD, uint32_t PPG, uint32_t MAX_INPUT_M>
inline void BMM_GEMV_a16_wfp8_block(
  uint8_t* input_data,
  uint8_t* weight_data,
  uint8_t* output_data,
  uint32_t M,
  uint32_t N,
  uint32_t K,
  uint32_t K_stride,
  uint32_t batch,
  float weight_scale,
  sycl::queue& q) {
    // Limitation:   K % 128 == 0   M <= MAX_INPUT_M    K % scale_block_size_K == 0 or K
    // NT <= 64    IT is fp16    K % HD == 0    HD <= 1024    PPG <= 1024     all hyper params must be 2^
    // batch here is for batch GEMV in absorb. indicated as head.
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

      const IT *  input_ptr = ((IT*)input_data) + hh * HD + b * K_stride;
      const uint8_t * weight_ptr = ((uint8_t *)weight_data) + h * PPG * K + hh * HD + b * N * K;
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
              input.template select<HD, 1>(HD * ii) = block_load<IT, HD>(input_ptr + CHUNK * ck + ii * batch * K_stride);
            }
          }

          // Loop PPG
          #pragma unroll
          for (int pp = 0; pp < PPG; pp++) {
            if (h < last_wg_id || (pp + PPG*h < N))  // only "not last wg" or "last wg but ppg in range" need execution
            {
                // read W   (HD)
                simd<uint8_t, HD> weight_quanted = block_load<uint8_t, HD>(weight_ptr + pp * K + CHUNK * ck);
                simd<IT, HD> weight{0};
                // FP8_E4M3_TO_FLOAT esimd impl
                {
                  simd<uint16_t, HD> result{0.0};
                  {
                    simd<uint8_t, HD> x = weight_quanted;
                    constexpr uint16_t weo = 5;
                    constexpr uint16_t wmo = 10;
                    // constexpr uint16_t ifNaN = 0x7F800001;
                  
                    auto is_zero = (x == 0);
                    // auto is_nan = (x == 0x80);
                  
                    simd<uint16_t, HD> mantissa = x & ((1 << WM) - 1);
                    simd<uint16_t, HD> exponent = (x & 0x7F) >> WM;

                    auto zero_exponent = (exponent == 0);
                    simd<uint16_t, HD> mantissa_subnormal = mantissa;
                    simd<uint16_t, HD> exponent_subnormal = exponent;
                    // subnormal input
                    {
                      // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above

                      simd<uint16_t, HD> renorm_shift_vec;
                      simd<uint16_t, HD> vec = mantissa;
                      {
                        simd<float, HD> vec_float = vec;

                        simd<uint32_t, HD> vec_uint = vec_float.template bit_cast_view<uint32_t>();
                        simd<uint32_t, HD> exponent_tmp = (vec_uint >> 23) & 0xFF;

                        simd<uint32_t, HD> lz = 158 - exponent_tmp; // 158 = 127 + 31

                        renorm_shift_vec = lz;
                      }
                      simd<uint16_t, HD> renorm_shift = renorm_shift_vec;

                      // renorm_shift111 = (IT)renorm_shift;

                      simd<uint16_t, HD> sh = 1 + renorm_shift - (32 - WM);
                      mantissa_subnormal <<= sh;
                      exponent_subnormal += 1 - sh;
                      mantissa_subnormal &= ((1 << WM) - 1);
                    }

                    mantissa.merge(mantissa_subnormal, zero_exponent);
                    exponent.merge(exponent_subnormal, zero_exponent);
                  
                    const uint16_t exp_low_cutoff = (1 << (weo - 1)) - (1 << (WE - 1));
                    exponent += exp_low_cutoff;
                    mantissa <<= wmo - WM;
                  
                    simd<uint16_t, HD> sign = x >> 7;
                    simd<uint16_t, HD> retval = (sign << 15) | (exponent << 10) | mantissa;

                    retval.merge(0, is_zero);
                    // retval.merge(ifNaN, is_nan);
              
                    result = retval;
                  }
              
                  weight = result.template bit_cast_view<IT>();
                }

                // dequant
                weight.template select<HD, 1>(0) = weight.template select<HD, 1>(0) * (IT)weight_scale;

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
            
            // writeOut  (PPG)
            block_store<IT, PPG>(((IT*)output_data) + h * PPG + b * M * N + ii * N, final_result);

          } // ii < M
        } // loop reduce_result_n_per_t
      } // check reduce_result_t_n

    });
  });
}
