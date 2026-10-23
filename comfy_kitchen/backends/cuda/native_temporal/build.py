"""Build the isolated temporal extension; never replaces Kitchen's main _C."""
import argparse,concurrent.futures,os,subprocess,shutil,hashlib,platform
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arch',default='120');p.add_argument('--jobs',type=int,default=2);p.add_argument('--build-dir',default='/tmp/ck-native-temporal-build');p.add_argument('--output');p.add_argument('--nvcc');p.add_argument('--compiler');args=p.parse_args()
if not args.arch.isdigit() or int(args.arch)<80:raise SystemExit('Use a numeric CUDA architecture, e.g. 120')
r=Path(__file__).resolve().parent;s=(r/'vendor') if (r/'vendor').is_dir() else r.parent/'sage_attention';b=Path(args.build_dir);b.mkdir(parents=True,exist_ok=True)
nvcc=args.nvcc or shutil.which('nvcc')
if not nvcc:raise SystemExit('nvcc is required; install a CUDA toolkit or use a matching prebuilt extension')
sources=[r/'sol_attn_temporal.cu',r/'sol_attn_exact_temporal.cu',r/'sage_temporal.cu',r/'native_error.cpp']+[s/f'sol_attn_{part}.cu' for part in ('preprocess','producer','vtranspose','route','token')]
windows=platform.system()=='Windows'
flags=[f'-DCK_TEMPORAL_ARCH={args.arch}','-DCOMFY_KITCHEN_TEMPORAL_EXTENSION=1','-DNDEBUG','-std=c++17','-O3','--use_fast_math','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode',f'arch=compute_{args.arch},code=sm_{args.arch}','-I'+str(s),'-I'+str(r)]
flags+=['-DNOMINMAX','-DWIN32_LEAN_AND_MEAN','-Xcompiler=/MD,/EHsc'] if windows else ['-Xcompiler=-fPIC']
if args.compiler:flags+=['-ccbin',args.compiler]
headers=b''.join(p.read_bytes() for p in sorted([*s.glob('*.cuh'),*s.glob('*.h'),*r.glob('*.h'),*r.glob('*.cuh')]))
compiler_version=subprocess.check_output([nvcc,'--version'])
if args.compiler:compiler_version+=subprocess.run([args.compiler] if windows else [args.compiler,'--version'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False).stdout
def compile_one(src):
 obj=b/(src.stem+('.obj' if windows else '.o'))
 stamp=obj.with_suffix('.sha256');digest=hashlib.sha256(src.read_bytes()+headers+repr(flags).encode()+compiler_version).hexdigest()
 if obj.exists() and stamp.exists() and stamp.read_text()==digest:
  print('cached',src.name,flush=True);return obj
 subprocess.run([nvcc,*flags,'-c',str(src),'-o',str(obj)],check=True)
 stamp.write_text(digest)
 print('compiled',src.name,flush=True);return obj
with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:objects=list(pool.map(compile_one,sources))
out=Path(args.output) if args.output else r/(f'libck_temporal_sm{args.arch}'+('.dll' if windows else '.so'))
temp=out.with_suffix(out.suffix+'.building')
subprocess.run([nvcc,*(['-ccbin',args.compiler] if args.compiler else []),'-shared',*(['-Xcompiler=/MD,/EHsc'] if windows else ['-Xcompiler=-fPIC','-Xlinker=-Bsymbolic']),*[str(x) for x in objects],'-o',str(temp)],check=True)
os.replace(temp,out)
print('built',out,flush=True)
