# SyclFFT Design Document

## Overview

A SYCL-native block-level FFT that replaces cuFFTDx in the GAMER ELBDM Gram-Fourier
Extension (GramFE) solver. It provides an in-place, batched, in-kernel
complex-to-complex FFT running entirely within a single SYCL work-group using
`sycl::local` memory.

The implementation lives in `include/SyclFFT.h` and is included by `include/FLU.h`
when `SYCL_LANGUAGE_VERSION` is defined and `GRAMFE_SCHEME == GRAMFE_FFT`.


## Replaced cuFFTDx API Surface

The former cuFFTDx typedef section in `include/FLU.h` (lines 546–599) was removed.
`SyclFFT.h` now defines all required symbols.

### Compile-time constants

| Symbol | Values | Source |
|---|---|---|
| `GRAMFE_FLU_NXT` | 64, 72, 108, 168, 300 | `Macro.h`: `FLU_NXT + GRAMFE_ND` |
| `gramfe_fft_float` | `float` or `double` | `Typedef.h` line 52 |
| `GRAMFE_NDELTA` | 14 | Gram polynomial extension |
| `elements_per_thread` | 4 | `GRAMFE_CUSTOM_ELEMENTS_PER_THREAD` |
| `ffts_per_block` | 12 | `GRAMFE_CUSTOM_FFTS_PER_BLOCK` |

### Types provided

| Type | Meaning |
|---|---|
| `complex_type` | `std::complex<gramfe_fft_float>` |
| `FFT::workspace_type` | Empty struct (no data needed) |
| `IFFT::workspace_type` | Same as `FFT::workspace_type` |
| `FFT::block_dim` | `syclfft::dim3` work-group shape |
| `FFT::max_threads_per_block` | Max threads per work-group |
| `FFT::shared_memory_size` | Shared memory in bytes for the FFT data only (cuFFTDx convention) |
| `FFT::ffts_per_block` | 12 |
| `FFT::elements_per_thread` | 4 |
| `FFT::value_type` | `std::complex<gramfe_fft_float>` |
| `FFT::size` | `GRAMFE_FLU_NXT` (replaces `cufftdx::size_of<FFT>::value`) |

### Methods provided

| Method | Signature |
|---|---|
| `FFT::execute(data, workspace)` | Run batched forward FFT on one work-group |
| `IFFT::execute(data, workspace)` | Run batched inverse FFT on one work-group |

---

## FFT Sizes

`GRAMFE_FLU_NXT` is a `#define` macro computed at compile time in `Macro.h`:

```cpp
#define GRAMFE_FLU_NXT  ( FLU_NXT + GRAMFE_ND )
```

Where `FLU_NXT = 2*PATCH_SIZE + 16` and `GRAMFE_ND` is selected by `PATCH_SIZE`:

| PATCH_SIZE | FLU_NXT | GRAMFE_ND | GRAMFE_FLU_NXT | Factorization |
|---|---|---|---|---|
| 8 | 32 | 32 | 64 | 2^6 |
| 16 | 48 | 24 | 72 | 2^3 * 3^2 |
| 32 | 80 | 28 | 108 | 2^2 * 3^3 |
| 64 | 144 | 24 | 168 | 2^3 * 3 * 7 |
| 128 | 272 | 28 | 300 | 2^2 * 3 * 5^2 |

All sizes are composite with small prime factors (2, 3, 5, 7), so a mixed-radix
Cooley-Tukey implementation handles all cases. No Rader's or Bluestein's algorithm
is needed.

---

## Execution Context

### Call sites

In `CPU_ELBDMSolver_GramFE_FFT.cpp`:

**Forward FFT:**
```cpp
FFT().execute(reinterpret_cast<void*>(s_In), Workspace);
```

**Inverse FFT:**
```cpp
IFFT().execute(reinterpret_cast<void*>(s_In), WorkspaceInv);
```

Both operate on `s_In`, a pointer to `complex_type[ffts_per_block][GRAMFE_FLU_NXT]`
in local memory.

### CPU reference path

```cpp
// Forward:
gramfe_fftw_c2c(FFTW_Plan_ExtPsi, s_In);

// Inverse:
gramfe_fftw_c2c(FFTW_Plan_ExtPsi_Inv, s_In);
```

`FFTW_Plan_ExtPsi` is created in `Init_FFTW.cpp`:
- Batch = `ffts_per_block` (12), Rank = 1, N = `GRAMFE_FLU_NXT`
- Sign = `FFTW_FORWARD`

Normalization: FFTW forward = scale 1, backward = scale 1/N.
The SYCL implementation uses the same unnormalized convention.

---

## Shared Memory Layout

### Layout

In `CPU_ELBDMSolver_GramFE_FFT.cpp` (SYCL path):

```cpp
sycl::local complex_type shared_mem[FFT::ffts_per_block * (GRAMFE_FLU_NXT + 2 * GRAMFE_NDELTA)];

complex_type (*s_In)[GRAMFE_FLU_NXT] = (complex_type (*)[GRAMFE_FLU_NXT]) (shared_mem);
complex_type (*s_Ae)[GRAMFE_NDELTA]  = (complex_type (*)[GRAMFE_NDELTA])  (shared_mem + CGPU_FLU_BLOCK_SIZE_Y * (GRAMFE_FLU_NXT));
complex_type (*s_Ao)[GRAMFE_NDELTA]  = (complex_type (*)[GRAMFE_NDELTA])  (shared_mem + CGPU_FLU_BLOCK_SIZE_Y * (GRAMFE_FLU_NXT + GRAMFE_NDELTA));
```

- `s_In`: FFT buffer — 12 rows of `GRAMFE_FLU_NXT` complex values
- `s_Ae`: Even Gram polynomial extension coefficients
- `s_Ao`: Odd Gram polynomial extension coefficients

The local array is sized in elements (not bytes), covering all three sub-arrays.

### Required size

In `GPU_Asyn_FluidSolver.cpp`:

```cpp
auto size       = FFT::ffts_per_block * FFT::size + 2 * FFT::ffts_per_block * GRAMFE_NDELTA;
auto size_bytes = size * sizeof(complex_type);
shared_size     = std::max(FFT::shared_memory_size, size_bytes);
```

---

## Workspace

### Creation

Workspace objects are default-constructed as empty structs in
`GPU_Asyn_FluidSolver.cpp` (no `cufftdx::make_workspace` call is needed since the
FFT operates entirely in local memory):

```cpp
FFT::workspace_type   workspace     = FFT::workspace_type{};
IFFT::workspace_type  workspace_inv = IFFT::workspace_type{};
```

---

## Block Dimensions and Threading

### Kernel launch

In `GPU_Asyn_FluidSolver.cpp`:

```cpp
cgh.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, NPatch_per_Stream[s]) * FFT::block_dim,
                      FFT::block_dim),
    [=](sycl::nd_item<3> item_ct1) {
       GPU_ELBDMSolver_GramFE_FFT(...);
    });
```

- Global range = `N_patches * block_dim` (each patch gets one work-group)
- Local range = `FFT::block_dim`, shape `{ ceil(N / elements_per_thread), ffts_per_block, 1 }`

### Thread indexing

```cpp
const uint tx  = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
const uint ty  = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(1);
const uint tid = ty * CGPU_FLU_BLOCK_SIZE_X + tx;
const uint NThread = CGPU_FLU_BLOCK_SIZE_X * CGPU_FLU_BLOCK_SIZE_Y;
```

where `CGPU_FLU_BLOCK_SIZE_X = FFT::block_dim.x` and
`CGPU_FLU_BLOCK_SIZE_Y = FFT::block_dim.y`.

### Data mapping

- `ffts_per_block` (12) concurrent FFTs, one per row of `s_In`
- Each thread handles `elements_per_thread` (4) elements per row via strided access
- Constraint: `elements_per_thread * total_threads >= ffts_per_block * GRAMFE_FLU_NXT`

The `block_dim` is computed as `{ ceil(N / elements_per_thread), ffts_per_block, 1 }`,
satisfying this constraint for all supported FFT sizes.

---

## GPU_Advance Function

The kernel entry `GPU_Advance` in `CPU_ELBDMSolver_GramFE_FFT.cpp` is shared between
CPU and SYCL paths via `#ifdef SYCL_LANGUAGE_VERSION`. The `FFT().execute()` and
`IFFT().execute()` calls slot into the existing call pattern without refactoring.

---

## Migration Notes

All work items from the original plan have been completed:

1. **`include/SyclFFT.h`** created — defines `complex_type`, `workspace`, `dim3`,
   `BlockFFT<T, N, Direction, FFTsPerBlock, ElementsPerThread>`, and GAMER aliases
   (`FFT`, `IFFT`, `complex_type`).
2. **Block FFT implemented** — batched, in-place, mixed-radix (4/2/3/5/7)
   decimation-in-time Cooley-Tukey operating on `sycl::local` memory.
3. **`include/FLU.h`** updated — `#include "SyclFFT.h"` replaces the cuFFTDx
   typedef section (former lines 546–599).
4. **`src/GPU_API/GPU_Asyn_FluidSolver.cpp`** updated — `cufftdx::make_workspace`
   replaced by empty `FFT::workspace_type{}`, `cufftdx::size_of<FFT>::value` replaced
   by `FFT::size`, `FFT::block_dim` used for launch.
5. **`src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_GramFE_FFT.cpp`** updated —
   `sycl::local complex_type shared_mem[N]` with element-count sizing,
   `FFT::workspace_type`/`IFFT::workspace_type` aliases, ambiguous `operator*`
   overloads removed (std::complex operators used directly).
6. **`src/GPU_API/GPU_SetCache.cpp`** updated — cuFFTDx shared-memory size queries
   replaced with `PreferShared` TODO comments.

All anticipated issues were resolved during implementation:

- **Ambiguous `operator*`**: The template helper operators were removed from the
  SYCL path; `std::complex` built-in operators are used directly.
- **`std::complex` on device**: DPC++ supports `std::complex` in kernels.
- **Local memory sizing**: Changed from byte-count (`FFT::shared_memory_size`) to
  element-count array declaration: `sycl::local complex_type shared_mem[ffts_per_block
  * (GRAMFE_FLU_NXT + 2 * GRAMFE_NDELTA)]`.
- **Cleanup**: `#include <cufftdx.hpp>` removed, `make_workspace` calls removed,
  `__launch_bounds__` guarded by `#ifndef SYCL_LANGUAGE_VERSION`.

