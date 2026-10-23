// SPDX-License-Identifier: Apache-2.0
// Specialized Kitchen INT8 launcher: compact temporal bias, one attention launch.
#include "qk_int_sv_i8_cuda.cuh"
#include "native_error.h"
#include <algorithm>
#include <stdexcept>
#include <string>
namespace {
template <int HEAD_DIM, int CTA_K, MaskMode mask_mode, typename DTypeOut,
          bool fuse_fp32_probabilities = true, int CTA_Q = 128,
          typename Offset = uint32_t>
void launch_impl(int8_t *q, int8_t *k, int8_t *v, DTypeOut *o, float *q_scale,
                 float *k_scale, float *v_scale, const void *mask,
                 int64_t mask_stride_b, int64_t mask_stride_h,
                 int64_t mask_stride_q, int64_t mask_stride_k,
                 int mask_dtype_code, int qo_len, int kv_len,
                 int num_qo_heads, int num_kv_groups, Offset stride_bz_q,
                 int stride_seq_q, Offset stride_h_q, Offset stride_bz_k,
                 int stride_seq_k, Offset stride_h_k, Offset stride_bz_v,
                 Offset stride_h_v, int stride_d_v, Offset stride_bz_o,
                 int stride_seq_o, Offset stride_h_o, float sm_scale,
                 int batch_size, cudaStream_t stream, const float *mask_tile_bias) {
  // Tiling constants — must match sage_attention.py and dlpack_bindings.cpp.
  // D>=128 otherwise needs too many live FP32 output accumulators per thread.
  // A 16-row warp tile halves that accumulator set.
  constexpr int WARP_Q = HEAD_DIM >= 128 ? 16 : 32;
  constexpr int WARP_K = CTA_K;

  size_t smem_max =
      std::max(static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(int8_t) +
                                   CTA_K * HEAD_DIM * sizeof(int8_t) +
                                   CTA_K * HEAD_DIM * sizeof(int8_t)),
               static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(half)));

  auto kernel = qk_int_sv_i8_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float, false,
      DTypeOut, ComputeUnit::kCudaCore, mask_mode, false, true, false, false,
      fuse_fp32_probabilities, Offset>;

  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_max));
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "sage_attn failed to request " + std::to_string(smem_max) +
        " bytes of dynamic shared memory: " + cudaGetErrorString(error));
  }

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  kernel<<<grid, block, smem_max, stream>>>(
      q, k, v, o, nullptr, q_scale, k_scale, v_scale, nullptr, mask,
      mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
      mask_dtype_code, qo_len, kv_len, num_kv_groups, stride_bz_q,
      stride_seq_q, stride_h_q, stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_h_v, stride_d_v, stride_bz_o, stride_seq_o,
      stride_h_o, sm_scale, mask_tile_bias);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("sage_attn kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}


} // namespace
CK_TEMPORAL_EXPORT int ck_temporal_dense(const void* q,const void* k,const void* v,void* out,
        const void* qs,const void* ks,const void* vs,const float* temporal,
        const int64_t* m,float scale,uint64_t stream_ptr) {
    try {
        const int N=m[0],H=m[1],padded=m[2],cta=m[3],dtype=m[4];
        const int64_t head=(int64_t)N*128, batch=H*head;
        const int64_t vhead=(int64_t)128*padded,vbatch=H*vhead;
        auto qi=const_cast<int8_t*>((const int8_t*)q);
        auto ki=const_cast<int8_t*>((const int8_t*)k);
        auto vi=const_cast<int8_t*>((const int8_t*)v);
        auto qsi=const_cast<float*>((const float*)qs);
        auto ksi=const_cast<float*>((const float*)ks);
        auto vsi=const_cast<float*>((const float*)vs);
        auto stream=(cudaStream_t)stream_ptr;
#define RUN(CK,DT) launch_impl<128,CK,MaskMode::kTemporal,DT,true>(qi,ki,vi,(DT*)out,qsi,ksi,vsi,temporal,0,0,0,0,-2,N,N,H,1,batch,128,head,batch,128,head,vbatch,vhead,padded,batch,128,head,scale,1,stream,nullptr)
        if (dtype==1) {if(cta==128) RUN(128,half); else RUN(64,half);}
        else {if(cta==128) RUN(128,nv_bfloat16); else RUN(64,nv_bfloat16);}
#undef RUN
        return 0;
    } catch(const std::exception& e) {ck_temporal_set_error(e.what());return -1;}
}
