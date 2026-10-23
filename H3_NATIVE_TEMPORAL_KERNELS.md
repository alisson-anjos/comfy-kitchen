# Experimental H3 temporal kernels for Comfy Kitchen

This branch adds an **optional, inference-only temporal overlay** for BFS MiniMax H3 native temporal alignment. It provides one-pass dense INT8 attention and one-pass sol-attn through native CUDA kernels. Ordinary Comfy Kitchen operators and its main `_C` extension are preserved.

Use BFSNodes [`main`](https://github.com/alisson-anjos/ComfyUI-BFSNodes/tree/main), version **1.115.22 or newer** for automatic multi-architecture downloads. The addon is separate from upstream Comfy Kitchen and is experimental. CUDA math and routing have been checked; full ComfyUI video quality and RTX 5090 end-to-end performance remain unverified.

## Automatic first-use setup in BFS 1.115.22+

On Linux x86_64 SM80/86/89/90/100/120 and Windows x86_64 SM86/89/120, with a driver supporting CUDA13, BFS 1.115.22+ downloads the pinned API and matching library automatically when Kitchen/sol-attn is first selected. The nine binaries passed CI compilation and native ABI checks; GPU execution on every architecture is not established. It verifies embedded SHA256 hashes and ABI/architecture, caches components in ComfyUI's user directory and activates them without pip, replacing Kitchen's main `_C`, or a second restart. A verified cache works offline. Failed/unsupported automatic setup retains the earlier backend with a console explanation. Existing compatible manual overlays are preferred.

See [BFS automatic setup documentation](https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/main/H3_NATIVE_TEMPORAL.md#automatic-first-use-setup-111517). The manual steps below remain available for explicit installs or source builds.

## Downloads

[Published Linux/Windows binaries and SHA256SUMS](https://github.com/alisson-anjos/comfy-kitchen/releases/tag/temporal-native-0413d43878606425a84aa400f1e5da50ed9dcdad). [GPU-specific direct download table (4090, H100, 5090 and more)](https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/main/H3_NATIVE_TEMPORAL.md#kernel-downloads).

A binary alone is not sufficient: the matching temporal Python API is required. Prefer BFS automatic setup, especially on Windows. The manual overlay installer below remains **Linux-only**. Confirm `Kitchen ONE-PASS dense kernel ACTIVE` or `Kitchen ONE-PASS sol kernel ACTIVE` in the sampling log; otherwise a compatibility path may be running.

## Install into the ComfyUI Python environment

Initial support: **Linux x86_64, CUDA FP16/BF16, batch 1, 128-wide heads**, native conditioning before the target video. A Linux SM120 build was tested on an RTX PRO 6000 Blackwell. RTX 5090 is also SM120, but that machine still needs local validation.

```bash
git clone --branch feat/h3-native-temporal https://github.com/alisson-anjos/comfy-kitchen.git comfy-kitchen-temporal
cd comfy-kitchen-temporal
/path/to/comfyui/python scripts/install_temporal_overlay.py --arch 120
```

The installer uses a matching local binary if present, otherwise explicitly builds the small temporal extension with `nvcc` (CUDA Toolkit supporting your SM; tested with CUDA 13). It does not build or replace Kitchen's main `_C`. No compilation occurs while sampling. Restart ComfyUI after installation.

A trusted prebuilt SM120 library can instead be downloaded from this fork's [experimental release](https://github.com/alisson-anjos/comfy-kitchen/releases/tag/h3-native-temporal-kernel-v1). Verify its checksum against the release's `SHA256SUMS`, then run:

```bash
/path/to/comfyui/python scripts/install_temporal_overlay.py --arch 120 --prebuilt /path/to/libck_temporal_sm120.so
```

The installer checks the native ABI and architecture. `--dry-run` prints the target without writing. `--target /path/to/site-packages/comfy_kitchen` can select an explicit existing package directory. Only `temporal_attention.py` and `backends/cuda/native_temporal/` are added/updated, using atomic file replacement. The latter includes source, vendor headers, a native library and an installation manifest. It does not alter the installed ordinary CUDA extension or its attention functions. To remove the addon, remove those two added paths with ComfyUI stopped.

## BFS usage

Use native temporal guide mode and route BFS's MODEL output to the guider. Identity references and the selected native video remain native; there is no Add Guide injection.

- **Dense Kitchen:** select `comfy kitchen attention` in Model Attention Backend, or explicitly select `comfy kitchen (experimental)` in BFS's native temporal backend.
- **sol-attn:** use the official Model Sparse Attention node with method `sol-attn`. It is detected automatically. Its normal start/end window, dense blocks and minimum-token threshold still determine when to use the sparse backend. SLA/VSA remain unsupported.

Example:

```text
LoRA -> Model Attention Backend (Kitchen dense fallback)
     -> Model Sparse Attention (optional, sol-attn)
     -> BFS native temporal conditioning -> guider
```

The BFS console confirms actual extension usage:

```text
Kitchen ONE-PASS dense kernel ACTIVE
Kitchen ONE-PASS sol kernel ACTIVE
```

Without a supported installed addon, BFS retains its existing compatibility implementations. Installing only a newer ordinary Comfy Kitchen package does not provide this temporal extension. A version/ABI/architecture mismatch raises an error rather than dispatching an incompatible kernel.

## What changed inside the kernels

For target query row `q` and selected guide key row `k`:

```text
bias(q,k) = -min(strength * (max(abs(t[q]-t[k])-window, 0)/falloff)^2, cap)
```

All other pairs retain zero temporal bias. Native temporal RoPE alignment still belongs to BFS; this extension computes only the bounded temporal score penalty on already projected/positioned Q/K/V.

A compact policy contains six row bounds/grid values, four penalty parameters and physical frame timestamps: **O(latent_frames)** storage. There is no NxN score/bias tensor and no loop launching a complete attention evaluation per frame. Configuration is passed per launch, with immutable Python policies and per-stream metadata buffers; no mutable GPU constant controls different models or streams.

### Dense INT8

The public Kitchen prequantization API quantizes Q/K/V once. A specialized native attention launch adds the temporal bias in the score/softmax path. Entirely unaffected tiles and tiles with one mathematically constant penalty use the fast INT8 probability path. Mixed-frame/boundary tiles compute the exact per-query/key penalty before online softmax. These shortcuts preserve the temporal rule; the underlying INT8 attention remains numerically approximate.

### sol-attn

The native Sol pipeline performs preprocess, routing/token augmentation and exact consumption **once**. The temporal bias is fused into its exact consumer using each MMA fragment's logical query/key indices, including the key-row permutation. All conditioning key blocks are forced exact because the pooled branch does not apply per-key bias. Other video keys retain Sol's sparse/pooled approximation. The addon therefore preserves the previous frame-wise Sol temporal policy without repeating the full sequence per frame.

This is one attention **pipeline**, not a claim of one CUDA kernel total: quantization, routing and token augmentation still use their normal stages. Current BFS integration uses post-RoPE Q/K/V and does not restore ComfyUI's streamed sparse QKV producer optimization.

## Measurements and validation

RTX PRO 6000 Blackwell, PyTorch 2.12.0+cu130, installed comfy-kitchen 0.2.35, concurrent Swap Anything training. Synthetic BF16 packed attention shape `[1,56,42836,128]`, 22 latent frames and 7,062 guide rows; one warmup plus one timed evaluation per backend:

| Implementation | Seconds |
| --- | ---: |
| Flash Partitions | 0.4232 |
| Previous frame-wise Kitchen dense | 0.5413 |
| **One-pass Kitchen dense** | **0.2463** |
| Previous frame-wise sol-attn | 2.6752 |
| **One-pass sol-attn** | **0.2163** |

These are isolated attention timings, not complete model/video or RTX 5090 timings. One-pass dense was about 1.7x faster than Flash Partitions in this diagnostic; one-pass Sol about 2.0x. Model offloading, projections, VAE and text encoding may limit end-to-end gains. Sol and dense attention also have different numerical/approximation behavior; faster does not establish visual quality.

GPU checks include FP16/BF16, partial spatial/frame tiles, nonuniform timestamps, zero strength/cap, large/small penalty caps, direct BFS entry points and simultaneous streams using different policies. In the small FP32-oracle cases dense INT8 relative output norm error was approximately 1.2–1.6%; forcing Sol's keys exact gave approximately 1.1–1.4%. One-pass Sol versus the same native pipeline repeated with frame-wise key biases differed by at most about **0.0022%** in the initial tested cases. The BFS dense and Sol bridges each made **one native API call**, and concurrent-stream outputs matched separate evaluations.

After installing, the small explicit GPU check can be reproduced with:

```bash
/path/to/comfyui/python scripts/verify_temporal_kernels.py --output verification.json
```

CPU policy tests check immutable configuration and invalid grid/parameter rejection. Diagnostic data is in [`docs/native_temporal/`](docs/native_temporal/). This is not a full generated-video validation. Existing checkpoint weights, optimizer state and live training backend were not modified by the addon.

## Source/build notes

- Python API: [`comfy_kitchen/temporal_attention.py`](comfy_kitchen/temporal_attention.py).
- Native sidecar: [`comfy_kitchen/backends/cuda/native_temporal/`](comfy_kitchen/backends/cuda/native_temporal/).
- Build: `python comfy_kitchen/backends/cuda/native_temporal/build.py --arch 120 --jobs 2`.
- Policies and carrier layouts are validated before raw-pointer launch. Native ABI v1 is architecture-checked; ordinary Kitchen's prequantized scale/stride contract is checked too.
- The extension uses CUDA runtime/native C interfaces rather than PyTorch's C++ ABI or `torch.compile`, so compiled kernels are shape-independent within the stated contract. A compatible GPU, Linux runtime and driver are still required.

Sources derive from Comfy Kitchen commit `f8b7e97a1418d2e529bd5405fbf86312be198ae6` and preserve original Apache-2.0 notices. The ordinary attention header additions are compile-gated by `COMFY_KITCHEN_TEMPORAL_EXTENSION`; normal Kitchen dispatch does not select the new mask mode. No training graph or model architecture/weights are changed.
