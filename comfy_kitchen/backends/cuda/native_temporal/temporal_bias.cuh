// SPDX-License-Identifier: Apache-2.0
// H3 temporal policy: compact metadata + physical frame timestamps, no NxN mask.
#pragma once
#include <cuda_runtime.h>
// p: sa,sb,ta,tb,source_frame_rows,target_frame_rows,strength,window,falloff,cap,times...
__device__ __forceinline__ float ck_temporal_bias(int q, int k, const float* p) {
    if (q < (int)p[2] || q >= (int)p[3] || k < (int)p[0] || k >= (int)p[1]) return 0.f;
    const int qf = (q-(int)p[2])/(int)p[5];
    const int kf = (k-(int)p[0])/(int)p[4];
    const float delta = fmaxf(fabsf(p[10+qf]-p[10+kf])-p[7],0.f)/p[8];
    return -fminf(p[6]*delta*delta,p[9]);
}
// An exact tile shortcut: use the fast INT8 probability path whenever the
// complete tile shares one bias (outside scope, one frame pair, zero window,
// or the capped far field). Boundary/mixed tiles keep the per-score rule.
__device__ __forceinline__ bool ck_temporal_tile_bias(int q0,int k0,int nq,int nk,const float* p,float& bias) {
    bias=0.f;
    if (p[6]==0.f || p[9]==0.f || q0+nq<=(int)p[2] || q0>=(int)p[3] || k0+nk<=(int)p[0] || k0>=(int)p[1]) return true;
    if (q0<(int)p[2] || q0+nq>(int)p[3] || k0<(int)p[0] || k0+nk>(int)p[1]) return false;
    int qa=(q0-(int)p[2])/(int)p[5],qb=(q0+nq-1-(int)p[2])/(int)p[5];
    int ka=(k0-(int)p[0])/(int)p[4],kb=(k0+nk-1-(int)p[0])/(int)p[4];
    if (qa==qb && ka==kb) {bias=ck_temporal_bias(q0,k0,p);return true;}
    float ql=p[10+qa],qh=p[10+qb],kl=p[10+ka],kh=p[10+kb];
    float maximum=fmaxf(fabsf(ql-kh),fabsf(qh-kl));
    if (maximum<=p[7]) return true;
    float minimum=fmaxf(fmaxf(ql-kh,kl-qh),0.f);
    float excess=fmaxf(minimum-p[7],0.f)/p[8];
    if (p[6]*excess*excess>=p[9]) {bias=-p[9];return true;}
    return false;
}
