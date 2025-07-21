#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

ESIMD_INLINE void residual_rmsNorm128PerThread_64t(uint8_t* weight, uint8_t* residual, uint8_t* hidden_states, uint8_t* hidden_states_out, int64_t hidden_size, int64_t input_len, int64_t add_residual, float variance_epsilon ,nd_item<1>& ndi) {

  __ESIMD_NS::slm_init(64 * sizeof(float));

  int h = ndi.get_group(0);
  int hh = ndi.get_local_linear_id();
  uint32_t inputOffset = 128 * hh * sizeof(fp16);
  int active_thread_num = hidden_size / 128;
  if (h >= input_len) return;

  simd<fp16, 128> input_FP16;
  simd<float, 128> input;
  simd<float, 128> input_powered;
  simd<fp16, 128> weight_FP16;
  simd<float, 16> variance = 0;
  simd<fp16, 128> residual_FP16;

  simd<float, 64> varianceSum = 0;
  int no_active_cnt = 64 - active_thread_num;

  if (hh < active_thread_num)
  {
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + h * hidden_size * sizeof(fp16) + inputOffset);
    
    if (add_residual == 1)
    {
      residual_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)residual + h * hidden_size * sizeof(fp16) + inputOffset);
    }

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset);
  }
  else
  {
    input_FP16 = 0;
    weight_FP16 = 0;
    residual_FP16 = 0;
  }

  if (add_residual == 1)
  {
    input = input_FP16 + residual_FP16;
  }
  else
  {
    input = input_FP16;
  }
  
  // write back new residual
  if (hh < active_thread_num)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)residual + h * hidden_size + 128 * hh, input.select<128, 1>(0));
  }

#pragma unroll
  for (int ll = 0; ll < 8; ll++) {
    input_powered.select<16, 1>(ll *16) = pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
    variance += input_powered.select<16, 1>(ll *16);
  }

  variance.select<8, 1>(0) += variance.select<8, 1>(8);
  variance.select<4, 1>(0) += variance.select<4, 1>(4);
  variance.select<2, 1>(0) += variance.select<2, 1>(2);
  variance[0] += variance[1];

  slm_block_store<float, 1>(hh * sizeof(float), variance[0]);

  barrier();
  
  varianceSum.select<64, 1>(0) = slm_block_load<float, 64>(0);

  for (int i = active_thread_num; i < 64; i++)
  {
    varianceSum[i] = 0;
  }
  
  varianceSum.select<32, 1>(0) += varianceSum.select<32, 1>(32);
  varianceSum.select<16, 1>(0) += varianceSum.select<16, 1>(16);
  varianceSum.select<8, 1>(0) += varianceSum.select<8, 1>(8);
  varianceSum.select<4, 1>(0) += varianceSum.select<4, 1>(4);
  varianceSum.select<2, 1>(0) += varianceSum.select<2, 1>(2);
  varianceSum[0] += varianceSum[1];

  varianceSum[0] = varianceSum[0] / hidden_size;
  varianceSum[0] =  sqrt(varianceSum[0] + variance_epsilon);

  simd<float, 128> varianceAVG;
  // fill 128 length varianceAVG
  varianceAVG[0] = varianceSum[0];
  varianceAVG[1] = varianceAVG[0];
  varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
  varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
  varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
  varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
  varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
  varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
  // simd<float, 128> varianceAVG{varianceSum[0]};

  input = input / varianceAVG;

  input_FP16 = input;
  input_FP16 = input_FP16 * weight_FP16;

  if (hh < active_thread_num)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + h * hidden_size + 128 * hh, input_FP16.select<128, 1>(0));
  }
}



ESIMD_INLINE void qk_rmsNorm128PerThread_32t(
  uint8_t* weight_q, uint8_t* weight_k, uint8_t* hidden_states_q, uint8_t* hidden_states_k,
  uint8_t* hidden_states_out_q, uint8_t* hidden_states_out_k,

  int64_t hidden_size_q,  // hidden_size
  int64_t hidden_size_k,  // hidden_size
  int64_t input_len,  // input len
  int64_t qk_stride,  // 2112
  float variance_epsilon_q,
  float variance_epsilon_k,
  nd_item<1>& ndi) {

  __ESIMD_NS::slm_init(32 * sizeof(float)); 

  int h = ndi.get_group(0);
  int hh = ndi.get_local_linear_id();
  int hh_q_n = hidden_size_q / 128;   // only support hidden_size_q == 1536 for now
  
  if (h >= input_len) return;

  simd<fp16, 128> input_FP16;
  simd<float, 128> input;
  simd<float, 128> input_powered;
  simd<fp16, 128> weight_FP16;
  simd<float, 16> variance = 0;
  simd<fp16, 128> residual_FP16;

  simd<float, 32> varianceSum = 0;

  int64_t q_stride = hidden_size_q;
  int64_t k_stride = hidden_size_k;
  if(qk_stride)
  {
    q_stride = k_stride = qk_stride;
  }

  if (hh < hh_q_n)
  {
    uint32_t inputOffset = 128 * hh * sizeof(fp16);
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states_q + h * q_stride * sizeof(fp16) + inputOffset);

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
	  __ESIMD_ENS::cache_hint::cached,
	  __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight_q + inputOffset);
  }
  else
  {
    uint32_t inputOffset = 128 * (hh - hh_q_n) * sizeof(fp16);
    input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states_k + h * k_stride * sizeof(fp16) + inputOffset);

    weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          256,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight_k + inputOffset);
  }

  input = input_FP16;

#pragma unroll
  for (int ll = 0; ll < 8; ll++) {
    input_powered.select<16, 1>(ll *16) = pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
    variance += input_powered.select<16, 1>(ll *16);
  }

  variance.select<8, 1>(0) += variance.select<8, 1>(8);
  variance.select<4, 1>(0) += variance.select<4, 1>(4);
  variance.select<2, 1>(0) += variance.select<2, 1>(2);
  variance[0] += variance[1];

  slm_block_store<float, 1>(hh * sizeof(float), variance[0]);

  barrier();
  
  varianceSum.select<32, 1>(0) = slm_block_load<float, 32>(0);
  if (hh < hh_q_n)
  {
    varianceSum.select<4, 1>(0) += varianceSum.select<4, 1>(4);
    varianceSum.select<2, 1>(0) += varianceSum.select<2, 1>(2);
    varianceSum[0] += varianceSum[1];
    varianceSum.select<2, 1>(8) += varianceSum.select<2, 1>(10);
    varianceSum[8] += varianceSum[9];
    varianceSum[0] += varianceSum[8];

    varianceSum[0] = varianceSum[0] / hidden_size_q;
    varianceSum[0] =  sqrt(varianceSum[0] + variance_epsilon_q);
  }
  else
  {
    varianceSum.select<2, 1>(12) += varianceSum.select<2, 1>(14);
    varianceSum[0] = varianceSum[12] + varianceSum[13];
    varianceSum[0] = varianceSum[0] / hidden_size_k;
    varianceSum[0] =  sqrt(varianceSum[0] + variance_epsilon_k);
  }

  simd<float, 128> varianceAVG;
  // fill 128 length varianceAVG
  varianceAVG[0] = varianceSum[0];
  varianceAVG[1] = varianceAVG[0];
  varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
  varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
  varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
  varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
  varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
  varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
  // simd<float, 128> varianceAVG{varianceSum[0]};

  input = input / varianceAVG;

  input_FP16 = input;
  input_FP16 = input_FP16 * weight_FP16;

  if (hh < hh_q_n)
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out_q + h * hidden_size_q + 128 * hh, input_FP16.select<128, 1>(0));
  }
  else
  {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out_k + h * hidden_size_k + 128 * (hh - hh_q_n), input_FP16.select<128, 1>(0));
  }
}
