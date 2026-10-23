"""Add only temporal API/kernels to an existing Kitchen install; preserve _C."""
import argparse,ctypes,hashlib,importlib.util,json,os,platform,shutil,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',type=Path,help='Existing comfy_kitchen package directory (normally detected from this Python environment)');p.add_argument('--arch',help='CUDA SM architecture, e.g. 120');p.add_argument('--prebuilt',type=Path,help='Use a trusted matching prebuilt .so instead of compiling');p.add_argument('--jobs',type=int,default=2);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):raise SystemExit('Initial native temporal overlay supports Linux x86_64 only')
root=Path(__file__).resolve().parents[1];source=root/'comfy_kitchen';native=source/'backends/cuda/native_temporal'
if args.target:target=args.target.resolve()
else:
 spec=importlib.util.find_spec('comfy_kitchen')
 if spec is None or spec.origin is None:raise SystemExit('Run this installer with the Python environment containing Comfy Kitchen')
 target=Path(spec.origin).resolve().parent
if target==source.resolve():raise SystemExit('Target resolves to this source checkout; use the ComfyUI Python environment or pass --target')
if not target.is_dir():raise SystemExit('Target must be an existing comfy_kitchen package directory')
arch=args.arch
if arch is None:
 import torch
 if not torch.cuda.is_available():raise SystemExit('Pass --arch when building without a visible CUDA GPU')
 major,minor=torch.cuda.get_device_capability();arch=f'{major}{minor}'
if not arch.isdigit() or int(arch)<80:raise SystemExit('Unsupported CUDA architecture')
binary=args.prebuilt.resolve() if args.prebuilt else native/f'libck_temporal_sm{arch}.so'
print(json.dumps({'target':str(target),'arch':arch,'binary':str(binary),'build_required':not binary.is_file(),'existing_Kitchen_core':'preserved'},indent=2))
if args.dry_run:raise SystemExit(0)
if args.prebuilt and not binary.is_file():raise SystemExit('Prebuilt binary not found')
if not binary.is_file():subprocess.run([sys.executable,str(native/'build.py'),'--arch',arch,'--jobs',str(args.jobs)],check=True)
lib=ctypes.CDLL(str(binary));lib.ck_temporal_abi_version.restype=ctypes.c_int;lib.ck_temporal_arch.restype=ctypes.c_int
if lib.ck_temporal_abi_version()!=1 or lib.ck_temporal_arch()!=int(arch):raise SystemExit('Prebuilt ABI/architecture does not match this installation')
dest=target/'backends/cuda/native_temporal';files=[(source/'temporal_attention.py',target/'temporal_attention.py')]
files += [(f,dest/f.name) for f in native.iterdir() if f.suffix in ('.py','.cpp','.cu','.h','.cuh')]
files += [(f,dest/'vendor'/f.name) for f in (source/'backends/cuda/sage_attention').iterdir() if f.suffix in ('.cu','.cuh','.h')]
files.append((binary,dest/f'libck_temporal_sm{arch}.so'))
files += [(root/name,dest/name) for name in ('LICENSE','NOTICE') if (root/name).is_file()]
manifest=[]
for src,dst in files:
 dst.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=dst.parent,prefix=dst.name+'.',delete=False) as f:temp=Path(f.name)
 try:
  shutil.copy2(src,temp);os.replace(temp,dst)
 finally:
  if temp.exists():temp.unlink()
 manifest.append({'path':str(dst.relative_to(target)),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
(dest/'installation.json').write_text(json.dumps({'arch':arch,'files':manifest},indent=2)+'\n')
print('Temporal overlay installed. Existing Kitchen _C and ordinary operators were not replaced. Restart ComfyUI.')
