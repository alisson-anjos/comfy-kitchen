"""CPU-only Windows compilation/ABI check; no model downloads or GPU needed."""
import ctypes,hashlib,json,os,shutil,subprocess,sys,tempfile,urllib.request,zipfile
from pathlib import Path
arch=int(os.environ.get('TEMPORAL_CUDA_ARCH','120'))
base='https://developer.download.nvidia.com/compute/cuda/redist/'
packages=[('cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-13.0.88-archive.zip','61760e24a44937c6c29a20ddea69067b4d3860ccd6a9a0564573f840d96e6c5f'),('cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.0.96-archive.zip','a2ed875f9997aa24904fb70cc9db3acd9308433cde99bc8e63ec1271c9da31b4'),('cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-13.0.85-archive.zip','f4f5187f7976d8057d76e1e04e72c8b7d0a2a22793107a671ad709b1d8d3408d')]
packages += [('cuda_crt/windows-x86_64/cuda_crt-windows-x86_64-13.0.88-archive.zip','ff413c89dbd96ed3b3dcfac923437de78acea32fb90309277197c8a127588112'),('libnvvm/windows-x86_64/libnvvm-windows-x86_64-13.0.88-archive.zip','6dda4a82d22a2c173a65c66b4b4c933e3424a56ce845e2864bdca9cc66e6c524')]
root=Path(__file__).resolve().parents[1];cuda=Path(os.environ['RUNNER_TEMP'])/'cuda-temporal';cuda.mkdir(exist_ok=True)
for relative,expected in packages:
 archive=cuda/Path(relative).name
 urllib.request.urlretrieve(base+relative,archive)
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   parts=Path(info.filename).parts[1:]
   if not parts or info.is_dir():continue
   assert '..' not in parts
   target=cuda.joinpath(*parts);target.parent.mkdir(parents=True,exist_ok=True)
   with z.open(info) as source,target.open('wb') as out:shutil.copyfileobj(source,out)
vswhere=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
vs=subprocess.check_output([str(vswhere),'-utf8','-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True,encoding='utf-8').strip().lstrip('\ufeff')
vcvars=Path(vs)/'VC/Auxiliary/Build/vcvars64.bat'
with tempfile.TemporaryDirectory(prefix='ck-msvc-') as temp:
 batch=Path(temp)/'environment.cmd'
 batch.write_text('@echo off\nchcp 65001 >nul\ncall "'+str(vcvars)+'" >nul\nif errorlevel 1 exit /b 1\nset\n',encoding='utf-8')
 output=subprocess.check_output([os.environ.get('COMSPEC','cmd.exe'),'/d','/c',str(batch)],text=True,encoding='utf-8',errors='replace')
env={k.upper():v for k,v in os.environ.items()}
for line in output.splitlines():
 if '=' in line and not line.startswith('='):
  k,v=line.split('=',1);env[k.upper()]=v
compiler=shutil.which('cl.exe',path=env['PATH']);assert compiler
nvcc=cuda/'bin/nvcc.exe';assert nvcc.is_file()
env['CUDA_PATH']=str(cuda);env['PATH']=str(cuda/'bin')+';'+env['PATH'];env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
native=root/'comfy_kitchen/backends/cuda/native_temporal';dll=root/f'libck_temporal_sm{arch}.dll'
subprocess.run([sys.executable,str(native/'build.py'),'--arch',str(arch),'--jobs','1','--build-dir',str(cuda/'objects'),'--output',str(dll),'--nvcc',str(nvcc),'--compiler',compiler],env=env,check=True)
lib=ctypes.CDLL(str(dll));assert lib.ck_temporal_abi_version()==1;assert lib.ck_temporal_arch()==arch
(root/'windows_abi.json').write_text(json.dumps({'native_abi':1,'arch':arch,'sha256':hashlib.sha256(dll.read_bytes()).hexdigest(),'bytes':dll.stat().st_size,'gpu_runtime_validation':False,'platform':'Windows','cuda':'13.0','source_commit':os.environ.get('GITHUB_SHA')},indent=2))
print('Windows DLL build and CPU ABI checks passed')
