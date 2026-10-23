"""CPU-only Linux CUDA build/ABI check from official redistributables."""
import ctypes,hashlib,json,os,shutil,subprocess,sys,tarfile,tempfile,urllib.request
from pathlib import Path
arch=int(os.environ.get('TEMPORAL_CUDA_ARCH','120'));base='https://developer.download.nvidia.com/compute/cuda/redist/'
meta=json.load(urllib.request.urlopen(base+'redistrib_13.0.2.json'))
root=Path(__file__).resolve().parents[1];cuda=Path(os.environ['RUNNER_TEMP'])/'cuda-temporal';cuda.mkdir(exist_ok=True)
for name in ('cuda_nvcc','cuda_cudart','cuda_cccl','cuda_crt','libnvvm'):
 package=meta[name]['linux-x86_64'];archive=cuda/Path(package['relative_path']).name
 urllib.request.urlretrieve(base+package['relative_path'],archive)
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==package['sha256']
 with tempfile.TemporaryDirectory() as temp:
  with tarfile.open(archive) as tf:tf.extractall(temp,filter='data')
  source=next(Path(temp).iterdir());shutil.copytree(source,cuda,dirs_exist_ok=True)
if not (cuda/'lib64').exists() and (cuda/'lib').exists():(cuda/'lib64').symlink_to('lib',target_is_directory=True)
env=dict(os.environ);env['PATH']=str(cuda/'bin')+os.pathsep+env['PATH'];env['CUDA_HOME']=str(cuda)
env['LD_LIBRARY_PATH']=os.pathsep.join([str(cuda/'lib64'),str(cuda/'lib'),str(cuda/'nvvm/lib64'),env.get('LD_LIBRARY_PATH','')])
native=root/'comfy_kitchen/backends/cuda/native_temporal';binary=root/f'libck_temporal_sm{arch}.so'
subprocess.run([sys.executable,str(native/'build.py'),'--arch',str(arch),'--jobs','1','--build-dir',str(cuda/'objects'),'--output',str(binary),'--nvcc',str(cuda/'bin/nvcc'),'--compiler',shutil.which('g++')],env=env,check=True)
lib=ctypes.CDLL(str(binary));assert lib.ck_temporal_abi_version()==1 and lib.ck_temporal_arch()==arch
(root/'linux_abi.json').write_text(json.dumps({'native_abi':1,'arch':arch,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'bytes':binary.stat().st_size,'gpu_runtime_validation':False,'platform':'Linux','cuda':'13.0','source_commit':os.environ.get('GITHUB_SHA')},indent=2))

# Exercise manual overlay installation without a GPU or an existing Kitchen core.
with tempfile.TemporaryDirectory(prefix='ck-overlay-check-') as temp:
    target=Path(temp)/'comfy_kitchen';target.mkdir()
    sentinel=target/'_C.original';sentinel.write_bytes(b'ordinary Kitchen core must remain unchanged')
    original=hashlib.sha256(sentinel.read_bytes()).hexdigest()
    subprocess.run([sys.executable,str(root/'scripts/install_temporal_overlay.py'),'--target',str(target),'--arch',str(arch),'--prebuilt',str(binary)],env=env,check=True)
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest()==original
    installed=target/'backends/cuda/native_temporal'/binary.name
    assert installed.is_file() and (target/'temporal_attention.py').is_file()
    assert hashlib.sha256(installed.read_bytes()).hexdigest()==hashlib.sha256(binary.read_bytes()).hexdigest()
print('Manual overlay installation passed without replacing the original Kitchen core')
