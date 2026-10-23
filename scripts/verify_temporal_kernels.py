import importlib.util,sys,json,time,torch
from pathlib import Path
# Run with the ComfyUI Python environment after installing the temporal overlay.
# Do not accidentally import the unbuilt source checkout's ordinary _C.
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('temporal_kernel_verification.json'));args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=root]
from comfy_kitchen import temporal_attention as native
report={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'cases':[]}
try:
 torch.manual_seed(34)
 for dtype in (torch.bfloat16,torch.float16):
  for frames,sf,tf,prefix,suffix in [(3,5,7,9,3),(7,64,96,128,0)]:
   sa=prefix;sb=sa+frames*sf;ta=sb+5;tb=ta+frames*tf;n=tb+suffix
   times=tuple(i*17/24 for i in range(frames))
   policy=native.TemporalPolicy(sa,sb,ta,tb,sf,tf,times)
   zero=native.TemporalPolicy(sa,sb,ta,tb,sf,tf,times,strength=0.)
   q,k,v=[torch.randn(1,2,n,128,device='cuda',dtype=dtype) for _ in range(3)]
   ts=torch.zeros(n,device='cuda');ts[sa:sb]=torch.tensor(times,device='cuda').repeat_interleave(sf);ts[ta:tb]=torch.tensor(times,device='cuda').repeat_interleave(tf)
   dt=(ts[:,None]-ts[None,:]).abs();bias=torch.zeros(n,n,device='cuda');bias[ta:tb,sa:sb]=-(2*(torch.relu(dt[ta:tb,sa:sb]-.2)/.25).square()).clamp(max=6.)
   expected=torch.nn.functional.scaled_dot_product_attention(q.float(),k.float(),v.float(),attn_mask=bias[None,None])
   dense=native.int8_attention(q,k,v,policy)
   exact=native.sol_attn(q,k,v,policy,token_aug=0,_exact_all_keys=True)
   actual=native.sol_attn(q,k,v,policy,tau=1.3,token_aug=256)
   repeated=native.sol_attn(q,k,v,zero,tau=1.3,token_aug=256).clone()
   for frame in range(frames):
    kb=bias[ta+frame*tf]
    part=native.sol_attn(q,k,v,zero,tau=1.3,token_aug=256,_fixed_key_bias=kb)
    repeated[:,:,ta+frame*tf:ta+(frame+1)*tf]=part[:,:,ta+frame*tf:ta+(frame+1)*tf]
   torch.cuda.synchronize()
   rel=lambda a,b:float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-10))
   row={'dtype':str(dtype),'shape':list(q.shape),'dense_vs_fp32':rel(dense,expected),'sol_exact_vs_fp32':rel(exact,expected),'sol_onepass_vs_framewise':rel(actual,repeated),'finite':bool(torch.isfinite(dense).all() and torch.isfinite(actual).all())}
   report['cases'].append(row);print(row,flush=True)
   assert row['finite'] and row['dense_vs_fp32']<.06 and row['sol_exact_vs_fp32']<.06 and row['sol_onepass_vs_framewise']<.02
 report['status']='passed'
except Exception as e:report.update(status='error',error=repr(e));print(repr(e),flush=True)
args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

raise SystemExit(0 if report['status']=='passed' else 1)
