# SYCL Migration Notes

This document preserves the architectural philosophy and key design decisions
from the CUDA-to-SYCL migration of GAMER. The migration was performed using
Intel's DPC++ Compatibility Tool (`dpct`) for an initial pass, followed by
extensive manual refinement. The `dpct_out/` directory (formerly at the
repository root) and the `MIGRATION_TRACKER.md` file have been removed from
version control; the knowledge they contained is consolidated here.

## Migration overview

GAMER originated as a CUDA-accelerated code for astrophysical adaptive
mesh refinement simulations. It has been migrated to **SYCL** (oneAPI
DPC++/Intel LLVM) as the sole GPU backend. The migration followed these
phases:

1. **Automated pass**: `dpct` was run against the original CUDA source
   tree, producing `dpct_out/` — a parallel directory structure of
   `.dp.cpp` suggestion files and `.yaml` replacement metadata.
2. **First pass**: Matching changes from `dpct_out/` were applied back
   into the original source tree (`src/`, `include/`).
3. **Second pass**: Manual refinement to resolve compilation issues,
   linker errors, and SYCL-specific semantics (sub-group sizes, kernel
   capture rules, host/device linkage, etc.).
4. **Verification**: ELBDM Gaussian wave packet test problem verified
   with identical results on both SYCL CPU and GPU execution.

## SYCL coding conventions

The following table documents the CUDA-to-SYCL transformation patterns
used consistently throughout the migrated codebase. These explain the
naming and idiom conventions you will encounter when reading the code.

### Compilation and dispatch guards

| CUDA | SYCL |
|---|---|
| `__CUDACC__` (compile guard) | `SYCL_LANGUAGE_VERSION` |
| `__global__` (function qualifier) | `SYCL_EXTERNAL` |
| `__device__` / `__forceinline__` | `GPU_DEVICE` (macro in `Macro.h`) |

### Thread hierarchy

| CUDA | SYCL |
|---|---|
| `blockIdx.x` | `get_nd_item<3>().get_group(2)` |
| `threadIdx.x` / `threadIdx.y` | `get_nd_item<3>().get_local_id(...)` |
| `__syncthreads()` | `sycl::group_barrier(sycl::nd_item<3>)` |
| `__shared__` | `sycl::local` (or `sycl::local_accessor` with pointer parameter) |
| CUDA thread blocks | SYCL work-groups |

### Memory management

| CUDA | SYCL |
|---|---|
| `cudaMalloc` | `sycl::malloc_device` |
| `cudaMallocHost` | `sycl::malloc_host` |
| `cudaFree` | `sycl::free` / `dpct_free` |
| `cudaFreeHost` | `sycl::free` |
| `cudaMemcpyToSymbol` | `dpct::get_in_order_queue().memcpy(...).wait()` |
| `cudaMemcpyFromSymbol` | Direct pointer assignment to device pointer |

### Function pointer transfers

Function pointers cannot be captured directly into SYCL kernels in the
way CUDA allows. The pattern used is to memcpy the host pointer to a
device pointer via a SYCL queue, then capture that device pointer by
value into the kernel lambda:

```cpp
// Host-side, before submit:
void* dev_ptr;
sycl::malloc_host(&dev_ptr, sizeof(func_ptr));
queue.memcpy(dev_ptr, host_ptr, sizeof(func_ptr)).wait();

// In kernel:
auto func_ptr = *static_cast<...>(dev_ptr);
```

### Error checking

- `DEVICE_CHECK_ERROR(expr)` catches `sycl::exception` directly and uses
  `e.what()` for the error message (replacing the DPCT-generated
  `dpct::get_error_string_dummy()` placeholder).
- `DEVICE_CHECK_MALLOC` catches `sycl::exception` from `sycl::malloc_*`
  calls.
- Both macros are wrapped in `do { ... } while(0)` to be safe in
  unbraced `if/else` chains.

## File naming conventions

Headers were renamed from CUDA-specific names to neutral, backend-agnostic
names:

| Legacy CUDA name | Current name |
|---|---|
| `CUDA_CheckError.h` | `CheckError.h` |
| `CUDA_ConstMemory.h` | `ConstMemory.h` |
| `CUAPI.h` | `GPUAPI.h` |
| `CUFLU.h` | `FLU.h` |
| `CUPOT.h` | `POT.h` |

The `dpct` tool's recommended `.dp.cpp` suffix was not used; migrated
files retain the standard `.cpp` extension.

## Function naming conventions

Device-side / GPU solver functions use a unified `GPU_` prefix, creating
clear symmetry with the `CPU_` CPU solvers:

| Prefix | Meaning | Examples |
|---|---|---|
| `GPU_` | Device-side solver functions | `GPU_FluidSolver_RTVD`, `GPU_PoissonSolver_MG` |
| `GPU_Asyn_` | Host-side asynchronous scheduler wrappers | `GPU_Asyn_FluidSolver`, `GPU_Asyn_PoissonGravitySolver` |
| `CPU_` | CPU fallback solvers | `CPU_FluidSolver_CTU`, `CPU_EoS_Gamma` |

Legacy prefixes (`CUFLU_`, `CUSRC_`, `CUPOT_`) have all been replaced
with the unified `GPU_` prefix.

## Cross-cutting architectural decisions

### 1. Compile-time dispatch (EoS and source terms)

SYCL kernels cannot use function pointers captured at runtime. The
original CUDA code used runtime function-pointer indirection for
equation-of-state selection and source-term dispatch. This was replaced
with **compile-time dispatch** via header-only macro systems:

- **`include/EoS_Dispatch.h`**: `EoSExternDecl` macro (`SYCL_EXTERNAL`
  under SYCL, `static` under CPU) and `CALL_EOS_*` macros that map to
  direct concrete-function calls under SYCL and to struct-member
  function-pointer calls under CPU.
- **`include/Src_Dispatch.h`**: Same pattern for source-term solvers
  (`Src_Deleptonization`, `Src_ExactCooling`, `Src_User_Template`).

This pattern is wired through `FLU.h` (which includes both dispatch
headers) and `src/SourceTerms/CPU_SrcSolver_IterateAllCells.cpp`.

### 2. Multiple-definition linker fix

Several EoS and source-term `.cpp` files (e.g. `CPU_EoS_Gamma.cpp`,
`CPU_Src_Deleptonization.cpp`) are compiled **twice** — once with
`-fsycl` (where functions are declared `SYCL_EXTERNAL` for external
device linkage) and once without (plain CPU host code). The
`EoSExternDecl` and `SrcExternDecl` macros expand to `static` (internal
linkage) under non-SYCL builds to prevent multiple-definition linker
errors. Under SYCL, they expand to `SYCL_EXTERNAL` (empty, preserving
external device linkage).

**Affected files**: all 5 EoS `.cpp` files and all 3 source-term `.cpp`
files listed in both `GPU_FILE` and `CPU_FILE` in `Makefile_base`.

### 3. Zero-length array safety

The codebase supports configurations where `NCOMP_PASSIVE == 0` (no
passive scalar variables). CUDA allowed zero-length array declarations
in some contexts, but this is undefined behavior in standard C++ and
triggers warnings or errors with strict SYCL compilers.

The fix wraps all conditionally-sized array declarations and struct
members in `MAX(1, NCOMP_PASSIVE)`, consistent with the existing pattern
already used for `NCOMP_MAG` and `NCOMP_ELE` in `Macro.h`:

```cpp
extern int VarIdx[MAX(1, NCOMP_PASSIVE)];   // instead of VarIdx[NCOMP_PASSIVE]
```

**Affected files**: `include/Global.h`, `include/HDF5_Typedef.h`,
`include/ConstMemory.h`, `src/Main/Main.cpp`,
`src/YT/YT_DerivedFunction.cpp`.

### 4. CPU sub-group size support (WARP_SIZE=32 enforcement)

The original CUDA code assumes `WARP_SIZE = 32` throughout. Intel CPUs
report a default sub-group size of 64, which caused runtime aborts. Two
complementary fixes were applied:

1. **Membership test instead of equality check**: Replaced
   `get_max_sub_group_size() != WARP_SIZE` (which rejected Intel CPUs
   because their max sub-group size is 64) with a membership test —
   does `WARP_SIZE` (32) appear in the device's supported sub-group
   sizes? This uses
   `dpct::get_current_device().get_info<sycl::info::device::sub_group_sizes>()`
   to query the full supported set directly from the SYCL device.
   Intel CPUs report `{8, 16, 32, 64}`, so 32 is supported even though
   it is not the maximum.

2. **`[[sycl::reqd_sub_group_size(32)]]`**: Added to `parallel_for`
   lambdas that use `WARP_SIZE`-dependent sub-group shuffles and
   warp-synchronous barriers (e.g. the SOR Poisson solver and
   `BlockReduction_Shuffle`/`BlockReduction_WarpSync` files). This
   forces 32-wide sub-groups on all devices regardless of the device
   default. The dt-solver kernels already had this attribute; the SOR
   solver was the only other kernel requiring it.

**Affected files**: `src/GPU_API/GPU_SetDevice.cpp`,
`src/GPU_API/GPU_Asyn_PoissonGravitySolver.cpp`.

### 5. Kernel global variable capture

SYCL forbids kernels from using non-const global variables directly.
The pattern used to fix this is to create a local `const` copy (or a
local non-const copy for mutable globals) inside the host-side
`submit()` callback before `cgh.parallel_for()`, then reference the
local in the `[=]` kernel lambda:

```cpp
cgh.submit([&](sycl::handler& cgh) {
    const EoS_t EoS_ct = EoS;          // local const copy
    auto GPUExtAcc_Ptr_ct = GPUExtAcc_Ptr;  // local copy

    cgh.parallel_for(..., [=](sycl::nd_item<3> item) {
        // use EoS_ct, GPUExtAcc_Ptr_ct here
    });
});
```

## SyclFFT

The cuFFTDx dependency (CUDA-specific) was replaced with a
SYCL-native block-level FFT implementation in `include/SyclFFT.h`. This
is a fully validated mixed-radix Cooley-Tukey implementation supporting
FFT sizes 64, 72, 108, 168, and 300. See `doc/SyclFFT/DESIGN.md` for the
complete design document, API surface, execution context, and validation
details.

## cuFLU / cuPOT reference

A mapping of original CUDA solver file names (`CUFLU_*`, `CUPOT_*`) to
their SYCL-era equivalents is maintained in `cuflu-cupot-notes.txt` at
the repository root. This file is kept as a reference for cross-
referencing against upstream CUDA forks but is not part of the
SYCL build.
