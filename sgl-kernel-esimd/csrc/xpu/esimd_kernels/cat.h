#include "utils.h"



// fp16, 512, 64
template <typename IT, uint32_t QK_DIM1, uint32_t QK_DIM2>
inline void esimd_cat(
    uint8_t* q1,    //[header,seq,dim]
    uint8_t* q2,    //[seq,header,dim] + stride 192
    uint8_t* k1,    //[seq,header,dim]
    uint8_t* k2,    //[seq,header,dim] + stride 2112
    uint8_t* q,     //[seq,header,dim]
    uint8_t* k,     //[seq,header,dim]
    int64_t seq_len,  // input len
    int64_t q_header_num,
    int64_t k_header_num,
    int64_t q_stride,
    int64_t k_stride,
    sycl::queue& dpcpp_queue) {

    const size_t GS = 1;

    // (Pdb) p q_nope_out.shape
    // torch.Size([7, 128, 512])
    // (Pdb) p q_nope_out.stride()
    // (512, 3584, 1)
    // (Pdb) p q_pe.shape
    // torch.Size([7, 128, 64])
    // (Pdb) p q_pe.stride()
    // (24576, 192, 1)


    // (Pdb) p k_nope.shape
    // torch.Size([7, 1, 512])
    // (Pdb) p k_nope.stride()
    // (512, 512, 1)
    // (Pdb) p k_pe.shape
    // torch.Size([7, 1, 64])
    // (Pdb) p k_pe.stride()
    // (2112, 64, 1)


    const size_t q1_seq_stride = QK_DIM1;
    const size_t q1_head_stride = seq_len*QK_DIM1;

    const size_t q2_head_stride = q_stride;
    const size_t q2_seq_stride = q_stride*q_header_num;

    const size_t k1_head_stride = QK_DIM1;
    const size_t k1_seq_stride = QK_DIM1*k_header_num;

    const size_t k2_head_stride = k_stride;
    const size_t k2_seq_stride = k_stride*k_header_num;

    const size_t q_head_stride = QK_DIM1 + QK_DIM2;
    const size_t q_seq_stride = q_head_stride*q_header_num;  

    const size_t k_head_stride = QK_DIM1 + QK_DIM2;
    const size_t k_seq_stride = k_head_stride*k_header_num;

    sycl::range<3> global_size(seq_len, q_header_num + k_header_num,  GS);
    sycl::range<3> local_size(1, 1, GS);

#if 1
    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {

                const size_t seq_idx = item.get_group(0);
                const size_t head_idx = item.get_group(1);
                const size_t thread_idx = item.get_local_id(2);

                const IT * q1_ptr = (const IT *)q1 + seq_idx * q1_seq_stride + head_idx  * q1_head_stride;
                const IT * q2_ptr = (const IT *)q2 + seq_idx * q2_seq_stride + head_idx  * q2_head_stride;

                //output
                IT * q_ptr = (IT *)q + seq_idx * q_seq_stride + head_idx  * q_head_stride;

		        const size_t out_dim = QK_DIM1+QK_DIM2; 
                simd<IT, out_dim> q_out;
                simd<IT, out_dim> k_out;

                //cat q1 and q2 to q
                if(head_idx < q_header_num)
                {
                    q_out.template select<QK_DIM1,1>(0) = block_load<IT, QK_DIM1>(q1_ptr);
                    q_out.template select<QK_DIM2,1>(QK_DIM1) = block_load<IT, QK_DIM2>(q2_ptr);
                    block_store<IT, out_dim>(q_ptr, q_out);
                }
                else
                {
                    const IT * k1_ptr = (const IT *)k1 + seq_idx * k1_seq_stride + (head_idx - q_header_num)  * k1_head_stride;
                    const IT * k2_ptr = (const IT *)k2 + seq_idx * k2_seq_stride + (head_idx - q_header_num)  * k2_head_stride;
                    IT * k_ptr = (IT *)k + seq_idx * k_seq_stride + (head_idx - q_header_num)  * k_head_stride;
                    //cat k1 and k2 to k
                    k_out.template select<QK_DIM1,1>(0) = block_load<IT, QK_DIM1>(k1_ptr);
                    k_out.template select<QK_DIM2,1>(QK_DIM1) = block_load<IT, QK_DIM2>(k2_ptr);

                    block_store<IT, out_dim>(k_ptr, k_out);
                }

            });
      });
#endif

}
