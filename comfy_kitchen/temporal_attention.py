# SPDX-License-Identifier: Apache-2.0
"""Experimental one-pass H3 temporal attention for Comfy Kitchen CUDA.

B=1, D=128, native conditioning before the target, equal Q/K/V length.
No global GPU constants, no score-matrix allocation, no per-frame attention loop.
Build the isolated native extension explicitly; existing Kitchen _C is untouched.
"""
import ctypes as C
from dataclasses import dataclass, field
import math
import sys
from pathlib import Path
import torch

VERSION = 'h3_native_temporal_kernel_v1'
_LIBRARIES = {}


@dataclass(frozen=True)
class TemporalPolicy:
    source_start: int
    source_stop: int
    target_start: int
    target_stop: int
    source_frame_rows: int
    target_frame_rows: int
    frame_times: tuple
    strength: float = 2.
    window: float = .2
    falloff: float = .25
    cap: float = 6.
    _buffers: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        rows=(self.source_start,self.source_stop,self.target_start,self.target_stop,self.source_frame_rows,self.target_frame_rows)
        if any(isinstance(v,bool) or not isinstance(v,int) for v in rows):
            raise ValueError('Temporal row bounds/counts must be integers')
        if not (0<=self.source_start<self.source_stop<=self.target_start<self.target_stop<2**24):
            raise ValueError('Native temporal policy requires disjoint source-before-target segments below 2**24 rows')
        if min(self.source_frame_rows,self.target_frame_rows)<=0:
            raise ValueError('Frame row counts must be positive')
        object.__setattr__(self,'frame_times',tuple(float(v) for v in self.frame_times))
        n=len(self.frame_times)
        if not n or self.source_stop-self.source_start != n*self.source_frame_rows or self.target_stop-self.target_start != n*self.target_frame_rows:
            raise ValueError('Temporal frame grids must have equal frame counts and fixed spatial rows')
        if not all(math.isfinite(v) for v in self.frame_times) or any(a>=b for a,b in zip(self.frame_times,self.frame_times[1:])):
            raise ValueError('Physical frame timestamps must be finite and strictly increasing')
        if not all(math.isfinite(v) for v in (self.strength,self.window,self.falloff,self.cap)) or min(self.strength,self.window,self.cap)<0 or self.falloff<=0:
            raise ValueError('Invalid temporal penalty parameters')

    def tensor(self,device):
        stream=torch.cuda.current_stream(device)
        key=(str(device),stream.cuda_stream)
        if key not in self._buffers:
            values=[self.source_start,self.source_stop,self.target_start,self.target_stop,self.source_frame_rows,self.target_frame_rows,self.strength,self.window,self.falloff,self.cap,*self.frame_times]
            self._buffers[key]=torch.tensor(values,dtype=torch.float32,device=device)
        return self._buffers[key]


def _library(device):
    major,minor=torch.cuda.get_device_capability(device);arch=f'{major}{minor}'
    if arch not in _LIBRARIES:
        suffix='.dll' if sys.platform=='win32' else '.so'
        path=Path(__file__).parent/'backends/cuda/native_temporal'/(f'libck_temporal_sm{arch}'+suffix)
        if not path.is_file():
            raise RuntimeError(f'Native temporal extension missing for SM{arch}. Build comfy_kitchen/backends/cuda/native_temporal/build.py --arch {arch}; existing Kitchen attention is unaffected.')
        lib=C.CDLL(str(path))
        lib.ck_temporal_abi_version.restype=C.c_int
        lib.ck_temporal_arch.restype=C.c_int
        if lib.ck_temporal_abi_version()!=1 or lib.ck_temporal_arch()!=int(arch):
            raise RuntimeError('Native temporal extension ABI or GPU architecture mismatch; rebuild the overlay.')
        ptr=C.c_void_p;meta=C.POINTER(C.c_int64)
        lib.ck_temporal_last_error.restype=C.c_char_p
        lib.ck_temporal_sol_workspace.argtypes=[C.c_int]*3;lib.ck_temporal_sol_workspace.restype=C.c_int64
        lib.ck_temporal_sol.argtypes=[ptr]*7+[meta,C.c_float,C.c_float,C.c_uint64];lib.ck_temporal_sol.restype=C.c_int
        lib.ck_temporal_dense.argtypes=[ptr]*8+[meta,C.c_float,C.c_uint64];lib.ck_temporal_dense.restype=C.c_int
        _LIBRARIES[arch]=lib
    return _LIBRARIES[arch]


def is_available(device=None):
    if sys.platform not in ('linux','win32') or not torch.cuda.is_available():return False
    device=torch.device('cuda') if device is None else torch.device(device)
    if device.type!='cuda':return False
    major,minor=torch.cuda.get_device_capability(device)
    suffix='.dll' if sys.platform=='win32' else '.so'
    return (Path(__file__).parent/'backends/cuda/native_temporal'/(f'libck_temporal_sm{major}{minor}'+suffix)).is_file()


def _validate(q,k,v,policy):
    if q.ndim!=4 or q.shape!=k.shape or q.shape!=v.shape or q.shape[0]!=1 or q.shape[-1]!=128:
        raise ValueError('Native temporal CUDA attention requires matching [1,H,N,128] Q/K/V')
    if q.numel()>=2**32:
        raise ValueError('Packed sequence exceeds the native temporal carrier offset range')
    if not q.is_cuda or q.device!=k.device or q.device!=v.device:
        raise ValueError('Native temporal Q/K/V must share a CUDA device')
    if q.dtype not in (torch.float16,torch.bfloat16) or q.dtype!=k.dtype or q.dtype!=v.dtype:
        raise ValueError('Native temporal attention requires matching FP16/BF16 Q/K/V')
    if not isinstance(policy,TemporalPolicy) or policy.target_stop>q.shape[2]:
        raise ValueError('Temporal policy exceeds the packed sequence')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
        raise ValueError('Native temporal kernels are inference-only')


def _checked(lib,status):
    if status<0:
        raise RuntimeError(lib.ck_temporal_last_error().decode(errors='replace'))


def int8_attention(q,k,v,policy,*,scale=None):
    """Kitchen quantization once, followed by one dense temporally biased launch."""
    _validate(q,k,v,policy)
    from comfy_kitchen.sage_attention import prequantize_int8_attention
    with torch.cuda.device(q.device):
        lib=_library(q.device);p=prequantize_int8_attention(q,k,v,scale=scale)
        if p.original_head_dim!=128 or p.cta_k not in (64,128):
            raise RuntimeError('Unsupported Kitchen prequantized carrier layout')
        # Native Kitchen's public prequantized contract: contiguous BHND Q/K,
        # flattened [B*H*D,padded_N] V, per-thread Q/K scales.
        expected_qsc=(1,q.shape[1],((q.shape[2]+127)//128)*32)
        expected_ksc=(1,q.shape[1],((q.shape[2]+p.cta_k-1)//p.cta_k)*4)
        if tuple(p.q_scale.shape)!=expected_qsc or tuple(p.k_scale.shape)!=expected_ksc or p.v_scale.numel()!=q.shape[1]*128:
            raise RuntimeError('Kitchen quantization scale layout changed; update the temporal extension.')
        if not all(t.is_contiguous() for t in (p.q,p.k,p.v,p.q_scale,p.k_scale,p.v_scale)):
            raise RuntimeError('Kitchen carrier layout changed; refusing unsafe temporal kernel dispatch')
        out=torch.empty_like(q,memory_format=torch.contiguous_format)
        metadata=(C.c_int64*5)(q.shape[2],q.shape[1],p.v.shape[1],p.cta_k,1 if q.dtype==torch.float16 else 2)
        parameters=policy.tensor(q.device)
        args=[t.data_ptr() for t in (p.q,p.k,p.v,out,p.q_scale,p.k_scale,p.v_scale,parameters)]
        status=lib.ck_temporal_dense(*args,metadata,p.attention_scale,torch.cuda.current_stream(q.device).cuda_stream)
        _checked(lib,status)
        return out


def sol_attn(q,k,v,policy,*,tau=1.3,token_aug=256,sink_q=(0,0),_exact_all_keys=False,_fixed_key_bias=None):
    """One native Sol pipeline; temporal penalty fused in its exact consumer.

    All conditioning keys are exact sinks. The remaining video keys retain
    Sol's routing and pooled approximation. Private controls are test oracles.
    """
    _validate(q,k,v,policy)
    if not math.isfinite(tau) or not 0<=tau<=4:
        raise ValueError('sol-attn tau must be finite in [0,4]')
    if token_aug not in (0,64,128,192,256):raise ValueError('Invalid token augmentation budget')
    n=q.shape[2];blocks=(n+63)//64
    if len(sink_q)!=2 or not 0<=sink_q[0]<=sink_q[1]<=blocks:raise ValueError('Invalid dense query block interval')
    with torch.cuda.device(q.device):
        lib=_library(q.device);heads=q.shape[1]
        qn,kn,vn=[t.transpose(1,2) for t in (q,k,v)]
        # Kitchen producer uses aligned 16-byte loads; contiguous NHD is the
        # conservative fallback for unusual tensor views.
        qn,kn,vn=[t if t.stride(-1)==1 and t.data_ptr()%16==0 and all(t.shape[i]<=1 or t.stride(i)%8==0 for i in range(3)) else t.contiguous() for t in (qn,kn,vn)]
        size=lib.ck_temporal_sol_workspace(n,heads,token_aug);_checked(lib,size)
        workspace=torch.empty(size,dtype=torch.uint8,device=q.device)
        out=torch.empty((1,n,heads,128),dtype=q.dtype,device=q.device)
        parameters=policy.tensor(q.device)
        fixed=None
        if _fixed_key_bias is not None:
            if _fixed_key_bias.shape!=(n,) or _fixed_key_bias.device!=q.device or not torch.isfinite(_fixed_key_bias).all():
                raise ValueError('Invalid diagnostic key bias')
            fixed=(_fixed_key_bias.float()*math.log2(math.e)).contiguous()
        sink_end=blocks if _exact_all_keys else (policy.target_start+63)//64
        values=[n,heads,1 if q.dtype==torch.float16 else 0,sink_end,*sink_q,token_aug,*qn.stride()[:3],*kn.stride()[:3],*vn.stride()[:3]]
        metadata=(C.c_int64*len(values))(*values)
        status=lib.ck_temporal_sol(qn.data_ptr(),kn.data_ptr(),vn.data_ptr(),out.data_ptr(),workspace.data_ptr(),parameters.data_ptr(),None if fixed is None else fixed.data_ptr(),metadata,tau,128**-.5,torch.cuda.current_stream(q.device).cuda_stream)
        _checked(lib,status)
        return out.transpose(1,2)
