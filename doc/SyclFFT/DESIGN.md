# SyclFFT Design Document

## Overview

A SYCL-native block-level FFT to replace cuFFTDx in the GAMER ELBDM Gram-Fourier
Extension (GramFE) solver. Must provide an in-place, batched, in-kernel
complex-to-complex FFT running entirely within a single SYCL work-group using
`sycl::local` memory.


## cuFFTDx API Surface (to be replaced)

Defined in `include/FLU.h`, lines 546–599.

### Compile-time constants

| Symbol | Values | Source |
|---|---|---|
| `GRAMME_FLU_NXT` | 64, 72, 108, 168, 300 | `Macro.h`: `FLU_NXT + GRAMME_ND` |
| `gramfe_fft_float` | `float` or `double` | `Typedef.h` line 52 |
| `GRAMME_NDELTA` | 14 | Gram polynomial extension |
| `elements_per_thread` | 4 | `GRAMME_CUSTOM_ELEMENTS_PER_THREAD` |
| `ffts_per_block` | 12 | `GRAMME_CUSTOM_FFTS_PER_BLOCK` |

### Types to provide

| Type | Meaning |
|---|---|
| `complex_type` | Complex value type (`std::complex<gramfe_fft_float>`) |
| `FFT::workspace_type` | Opaque workspace (can be empty struct) |
| `IFFT::workspace_type` | Opaque workspace for inverse |
| `FFT::block_dim` | `sycl::range<3>` for work-group size |
| `FFT::max_threads_per_block` | Max threads per block |
| `FFT::shared_memory_size` | Min shared memory (elements) |
| `FFT::ffts_per_block` | = 12 |
| `FFT::elements_per_thread` | = 4 |
| `FFT::value_type` | = `complex_type` |

### Methods to provide

| Method | Signature |
|---|---|
| `FFT::execute(data, workspace)` | Run batched forward FFT on one work-group |
| `IFFT::execute(data, workspace)` | Run batched inverse FFT on one work-group |

---

## FFT Sizes

`GRAMME_FLU_NXT` is a `#define` macro computed at compile time:

```cpp
#define GRAMME_FLU_NXT  (FLU_NXT + GRAMME_ND)
```

Where `FLU_NXT = 2*PATCH_SIZE + 16` and `GRAMME_ND` is selected by `PATCH_SIZE`:

| PATCH_SIZE | FLU_NXT | GRAMME_ND | GRAMME_FLU_NXT | Factorization |
|---|---|---|---|---|
| 8 | 32 | 32 | 64 | 2^6 |
| 16 | 48 | 24 | 72 | 2^3 * 3^2 |
| 32 | 80 | 28 | 108 | 2^2 * 3^3 |
| 64 | 144 | 24 | 168 | 2^3 * 3 * 7 |
| 128 | 272 | 28 | 300 | 2^2 * 3 * 5^2 |

All sizes are composite with small prime factors (2, 3, 5, 7).
A mixed-radix Cooley-Tukey implementation handles all cases.
No Rader's or Bluestein's algorithm is needed.

---

## Execution Context

### Call sites

In `CPU_ELBDMSolver_GramFE_FFT.cpp`:

**Forward FFT (line ~549):**
```cpp
FFT().execute(reinterpret_cast<void*>(s_In), Workspace);
```

**Inverse FFT (line ~571):**
```cpp
IFFT().execute(reinterpret_cast<void*>(s_In), WorkspaceInv);
```

Both operate on `s_In`, a pointer to `complex_type[ffts_per_block][GRAMME_FLU_NXT]`
in shared/local memory.

### CPU reference path

```cpp
// Forward:
gramfe_fftw_c2c(FFTW_Plan_ExtPsi, s_In);

// Inverse:
gramfe_fftw_c2c(FFTW_Plan_ExtPsi_Inv, s_In);
```

`FFTW_Plan_ExtPsi` is created in `Init_FFTW.cpp`:
- Batch = `ffts_per_block` (12), Rank = 1, N = `GRAMME_FLU_NXT`
- Sign = `FFTW_FORWARD`

Normalization: FFTW forward = scale 1, backward = scale 1/N.
The SYCL implementation must match this (or match cuFFTDx's convention).

---

## Shared Memory Layout

### Layout

In `CPU_ELBDMSolver_GramFE_FFT.cpp` (lines 309–321):

```cpp
__shared__ complex_type  s_In[ffts_per_block][GRAMME_FLU_NXT];
__shared__ complex_type  s_Ae[CGPU_FLU_BLOCK_SIZE_Y][GRAMME_NDELTA];
__shared__ complex_type  s_Ao[CGPU_FLU_BLOCK_SIZE_Y][GRAMME_NDELTA];
```

- `s_In`: FFT buffer — 12 rows of `GRAMME_FLU_NXT` complex values
- `s_Ae`: Even Gram polynomial extension coefficients
- `s_Ao`: Odd Gram polynomial extension coefficients

### Required size

In `GPU_Asyn_FluidSolver.cpp` (lines 495–501):

```cpp
auto size       = ffts_per_block * size_of<FFT>::value + 2 * ffts_per_block * GRAM_ME_NDELTA;
auto size_bytes = size * sizeof(complex_type);
shared_size     = std::max(FFT::shared_memory_size, size_bytes);
```

The replacement must compute or hard-code `shared_memory_size` accordingly.

---

## Workspace

### Creation

In `GPU_Asyn_FluidSolver.cpp` (lines 840–846):

```cpp
FFT::workspace_type   workspace   = cufftdx::make_workspace<FFT>(error_code);
IFFT::workspace_type  workspace_inv = cufftdx::make_workspace<IFFT>(error_code);
```

**Replacement:** `workspace_type` can be an empty struct since the FFT operates
entirely in local memory:

```cpp
struct FFTWorkspace {};
using FFT_workspace_type   = FFTWorkspace;
using IFFT_workspace_type  = FFTWorkspace;
```

---

## Block Dimensions and Threading

### Kernel launch

In `GPU_Asyn_FluidSolver.cpp` (lines 853–861):

```cpp
sycl::nd_range<3>(
    sycl::range<3>(1, 1, NPatch_per_Stream[s]) * FFT::block_dim,  // global
    FFT::block_dim                                                // local
);
```

- Global range = `N_patches * block_dim` (each patch gets one work-group)
- Local range = `FFT::block_dim` (e.g., `{32, 4, 1}` = 128 threads)

### Thread indexing

In `CPU_ELBDMSolver_GramFE_FFT.cpp`:

```cpp
const uint tx = get_nd_item<3>().get_local_id(2);   // [0, block_dim.x)
const uint ty = get_nd_item<3>().get_local_id(1);   // [0, block_dim.y)
const uint tid = ty * CGPU_FLU_BLOCK_SIZE_X + tx;   // 1D thread ID
const uint NThread = CGPU_FLU_BLOCK_SIZE_X * CGPU_FLU_BLOCK_SIZE_Y;
```

### Data mapping

- `ffts_per_block` (12) concurrent FFTs, one per row of `s_In`
- Each thread handles `elements_per_thread` (4) elements per row via strided access:
  ```
  element_idx = tid * elements_per_thread + {0, 1, 2, 3}
  ```
- The `tid`-th thread with `elements_per_thread` elements must cover
  `elements_per_thread * NThread >= ffts_per_block * GRAMME_FLU_NXT`.

### Required block_dim

From cuFFTDx's algorithm configuration, the work-group must have enough threads
to satisfy `elements_per_thread * total_threads >= ffts_per_block * FFT_size`.
The exact `block_dim` should be extracted from cuFFTDx's `FFT::block_dim`
(see FLU.h for the SM capability and resulting block dimensions).

---

## GPU_Advance Function

The kernel entry `GPU_Advance` (in `CPU_ELBDMSolver_GramFE_FFT.cpp`, lines 470–634)
is shared between CPU and SYCL paths via `#ifdef SYCL_LANGUAGE_VERSION`.

The `FFT().execute()` and `IFFT().execute()` calls must either:
1. Slot into the existing call pattern, or
2. Be refactored to a different calling convention.

---

## Summary of Work Items

1. **Create `include/SyclFFT.h`** — Define `complex_type`, `FFTWorkspace`, constants,
   `block_dim`, `shared_memory_size`, and `execute()` methods.
2. **Implement the block FFT** — Batched, in-place, mixed-radix Cooley-Tukey FFT
   operating on `sycl::local` memory within a single work-group.
3. **Update `include/FLU.h`** — Replace the cuFFTDx typedef section (lines 546–599)
   with `#include "SyclFFT.h"`.
4. **Update `GPU_SetCache.cpp`** — Replace cuFFTDx shared-memory size queries
   (line ~108).
5. **Update `GPU_Asyn_FluidSolver.cpp`** — Remove cuFFTDx workspace creation
   (lines ~840–846).
6. **Compile and test** — Build under `SYCL_LANGUAGE_VERSION` and run.
