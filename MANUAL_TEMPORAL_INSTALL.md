# Manual TARA kernel installation

These instructions install the optional **one-pass attention extension**, not an entire H3/TARA application. They support the existing published CUDA13 v1 binaries. Stop ComfyUI before updating a manually installed DLL/library, and use **the same Python executable as ComfyUI**. Do not replace your normal Comfy Kitchen `_C` extension or install the whole fork over it.

## What is required without BFS?

BFSNodes is **not a mathematical requirement** for TARA. It is our ready-to-use ComfyUI integration. A different implementation must still:

1. Encode the motion video through H3's **native video reference** channel.
2. Align the selected motion reference's temporal RoPE clock to the generated video's clock.
3. Identify the output-video queries and motion-video keys, with matching physical timestamps.
4. Pass those bounds/timestamps and the temporal penalty settings to the temporal attention API.

The library by itself does not apply those steps. Loading a LoRA, selecting ordinary Kitchen attention, or copying a DLL into a folder does not automatically enable TARA. The BFS standalone TARA node can be used without the Planner; another application can call the Python API directly. See [Python API and policy](comfy_kitchen/temporal_attention.py) and [BFS model patch integration](https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/main/minimax_h3_native_temporal.py). Their upstream licenses/attributions still apply.

## Prerequisites and files

- Existing compatible Comfy Kitchen, PyTorch and H3 installation.
- 64-bit Linux/Windows on x86_64, matching NVIDIA architecture, driver supporting **CUDA13**. PyTorch's CUDA version can differ.
- A matching binary **plus `temporal_attention.py`** from the compatible source. Our installer copies that API, library, source/licenses and an installation manifest while preserving ordinary Kitchen operators.
- Linux binaries: SM80/86/89/90/100/120. Windows: SM86/89/120. RTX4090 is SM89; H100 is SM90 (Linux binary available); RTX5090 is SM120. Other “6000” cards may be SM86,89 or120.

[Release and SHA256SUMS](https://github.com/alisson-anjos/comfy-kitchen/releases/tag/temporal-native-0413d43878606425a84aa400f1e5da50ed9dcdad). [Direct per-GPU links](https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/main/H3_NATIVE_TEMPORAL.md#kernel-downloads). Those binaries have CI compilation/ABI checks, not GPU workflow validation on every card.

## Linux example (SM120 / RTX5090)

Edit the Python path and architecture. Keep the working directory outside the cloned source so Python finds the installed Kitchen package rather than the source checkout. No CUDA toolkit is needed when using a prebuilt.

```bash
tara_python=/path/to/ComfyUI/venv/bin/python
tara_work="$PWD/tara-kernels"
tara_arch=120 # 89 for RTX4090, 90 for H100
mkdir -p "$tara_work"
git clone --branch feat/h3-native-temporal https://github.com/alisson-anjos/comfy-kitchen.git "$tara_work/source"
tara_release=https://github.com/alisson-anjos/comfy-kitchen/releases/download/temporal-native-0413d43878606425a84aa400f1e5da50ed9dcdad
curl --fail --location "$tara_release/libck_temporal_sm${tara_arch}.so" --output "$tara_work/libck_temporal_sm${tara_arch}.so"
curl --fail --location "$tara_release/SHA256SUMS" --output "$tara_work/SHA256SUMS"
"$tara_python" - "$tara_work" "$tara_arch" <<'PY'
from pathlib import Path
import subprocess, sys
work, arch = Path(sys.argv[1]), sys.argv[2]
filename = f'libck_temporal_sm{arch}.so'
checksums = dict(line.split(None, 1)[::-1] for line in (work/'SHA256SUMS').read_text().splitlines())
checksums = {name.strip(): sha for name, sha in checksums.items()}
subprocess.run([sys.executable, str(work/'source/scripts/install_temporal_overlay.py'),
               '--arch', arch, '--prebuilt', str(work/filename), '--sha256', checksums[filename]], check=True)
PY
```

The installer checks the requested checksum **before loading the native library**, then checks ABI and architecture before copying. Download errors or absent checksum entries must be resolved; do not continue by removing verification. If the source directory already exists, inspect/update it instead of overwriting local work.

## Windows / ComfyUI Portable example (SM120 / RTX5090)

Run in PowerShell, outside the source checkout. Edit the ComfyUI Python path. `python_embeded` is the usual Portable spelling; use your actual installation. Change the architecture to89 for RTX4090 or86 for RTX30/A6000.

```powershell
$taraPython = 'C:\ComfyUI_windows_portable\python_embeded\python.exe'
$taraWork = Join-Path $env:USERPROFILE 'tara-kernels'
$taraArch = 120
New-Item -ItemType Directory -Force $taraWork | Out-Null
git clone --branch feat/h3-native-temporal https://github.com/alisson-anjos/comfy-kitchen.git "$taraWork\source"
$taraRelease = 'https://github.com/alisson-anjos/comfy-kitchen/releases/download/temporal-native-0413d43878606425a84aa400f1e5da50ed9dcdad'
$taraFile = "libck_temporal_sm${taraArch}.dll"
Invoke-WebRequest "$taraRelease/$taraFile" -OutFile "$taraWork\$taraFile"
Invoke-WebRequest "$taraRelease/SHA256SUMS" -OutFile "$taraWork\SHA256SUMS"
$taraLine = Get-Content "$taraWork\SHA256SUMS" | Where-Object { $_ -match ('  ' + [regex]::Escape($taraFile) + '$') }
if (@($taraLine).Count -ne 1) { throw 'Missing or ambiguous checksum; stop here.' }
$taraExpected = ($taraLine -split '\s+')[0]
& $taraPython "$taraWork\source\scripts\install_temporal_overlay.py" --arch $taraArch --prebuilt "$taraWork\$taraFile" --sha256 $taraExpected
```

The DLL requires the NVIDIA driver and compatible Windows/MSVC runtime. Source compilation additionally requires NVCC and the appropriate compiler environment; it is not required for this prebuilt route. Do not rename a Linux `.so` to `.dll` or use an SM120 binary on an SM89 GPU.

## Check installation and actual activation

From outside the source checkout, run with ComfyUI's Python:

```bash
/path/to/comfy/python -c "import comfy_kitchen.temporal_attention as t; print(t.VERSION); print(t.__file__); print('binary present for this GPU:', t.is_available())"
```

On Windows replace the executable with the quoted Portable path and prefix it with PowerShell's `&` invocation operator. `is_available()` checks that an appropriate library is present; it does **not** execute an attention kernel or prove end-to-end correctness.

Restart ComfyUI, then use a TARA-aware workflow/integration. With BFS look for `Kitchen ONE-PASS dense kernel ACTIVE` or `Kitchen ONE-PASS sol kernel ACTIVE` during sampling. Importing the module alone does not establish activation. Without BFS, the caller must wire the temporal policy and RoPE alignment explicitly.

For an explicit target package, use `--target /actual/path/to/site-packages/comfy_kitchen`. `--dry-run` reports paths without copying/loading; it does not validate the binary. To uninstall the manual overlay with ComfyUI stopped, remove its added `temporal_attention.py` and `backends/cuda/native_temporal/`, after checking `installation.json`. This does not remove any separate automatic BFS cache. Only remove those addon paths, not Kitchen's original `_C` or other operators.

## Source build

The build accepts an explicit SM, e.g. `python comfy_kitchen/backends/cuda/native_temporal/build.py --arch 89 --jobs 2`. Match the CUDA toolkit, host compiler and operating system. Windows source builds need the MSVC developer environment; they have not been made into an automatic end-user compiler installation. For numerical/runtime validation see `scripts/verify_temporal_kernels.py`; it needs an actual compatible GPU and consumes GPU resources.
