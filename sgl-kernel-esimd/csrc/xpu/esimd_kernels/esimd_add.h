#include "utils.h"
using bf16 = sycl::ext::oneapi::bfloat16;

template<typename TP>
ESIMD_INLINE void esimd_add_impl(uint8_t* a, uint8_t* b, uint8_t* c, int64_t len, float factor, nd_item<1>& ndi) {
  // __ESIMD_NS::slm_init(16 * sizeof(fp16));

  int global_idx = ndi.get_group(0);
  int local_range = ndi.get_local_range(0);
  int local_idx = ndi.get_local_id(0);
  int inputOffset = (global_idx * local_range + local_idx) * 128 * sizeof(fp16);
  simd<bf16, 128> a_bf16;
  simd<fp16, 128> b_fp16;
  // simd<fp16, 128> c_fp16;
  simd<TP, 128> c_float;

  a_bf16.template bit_cast_view<uint8_t>().template select<256, 1>(0) = __ESIMD_ENS::lsc_block_load<
      uint8_t,
      256,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + inputOffset);

  b_fp16.template bit_cast_view<uint8_t>().template select<256, 1>(0) = __ESIMD_ENS::lsc_block_load<
      uint8_t,
      256,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + inputOffset);

  a_bf16 = a_bf16 * factor;
  // c_fp16 = a_bf16 + b_fp16;
  c_float = a_bf16 + b_fp16;

  // if (local_idx < 16)
  {
    block_store<TP, 128>((TP*)c + 128 * (global_idx * local_range + local_idx), c_float);
    // __ESIMD_ENS::lsc_block_store<
    //     fp16,
    //     128,
    //     __ESIMD_ENS::lsc_data_size::default_size,
    //     __ESIMD_ENS::cache_hint::write_back,
    //     __ESIMD_ENS::cache_hint::write_back>(
    //     (fp16*)c + 128 * (global_idx * local_range + local_idx), c_fp16.select<128, 1>(0));
  }
  // barrier();
}
