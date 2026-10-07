# GAMER SYCL migration tracker

This file tracks which **original GAMER source files** have had the corresponding **DPCT suggestions from `dpct_out/`** applied back into the main codebase.

## Important context
- `dpct_out/` contains a duplicate directory structure for the repository.
- Files under `dpct_out/` are the **automatic migration output / suggestions**.
- The actual codebase to edit lives **outside** `dpct_out/`.
- At the start of tracking, no files outside `dpct_out/` had been migrated yet.
- A file not mentioned here is assumed not to have been migrated yet.

## How to use this tracker
- Mark a file here only after its matching change has been applied to the original file.
- Use this file as the source of truth for migration progress.
- Keep entries aligned with the paths in `dpct_out/` and the corresponding paths in the root source tree.
- **First-pass and Second-pass policy**:
  - The first pass is to attempt all files. Do not retry on the first pass if a file has complex compilation issues or blockages; move to the next file.
  - Retries, deeper debugging, and unresolved issues should be tackled on the second pass.

## Suggested status format
- `TODO` — not yet applied to the original file
- `IN PROGRESS` — partially applied or under manual review
- `DONE` — DPCT suggestion has been applied to the original file and verified
- `NOT STARTED` — file not yet addressed; still contains CUDA-specific constructs or un-migrated `__CUDACC__` dispatch guards

## Notes shorthand legend

Common long-form expressions in the Notes column are abbreviated below. Notes in the
tracker table use these shorthands for brevity; any text not covered by a shorthand
is preserved verbatim.

| Shorthand | Long form |
| --- | --- |
| `DPCT→SYCL` | Applied DPCT SYCL migration |
| `+hdrs` | added SYCL/dpct headers |
| `+inc` | added SYCL/dpct includes |
| `+shared` | added `real *shared` parameter to SYCL function signature |
| `C→S` | replaced `__CUDACC__` / `__CUDACC__` guards with `SYCL_LANGUAGE_VERSION` (incl. `CheckError.h` include guards) |
| `GAMER→Macro` | replaced `#include "GAMER.h"` with `#include "Macro.h"` (+ `POT.h`/`FLU.h`) |
| `g→EXT` | replaced `__global__` with `SYCL_EXTERNAL` |
| `d→DEV` | replaced `__device__`/`__forceinline__` with `GPU_DEVICE` |
| `bx→grp(2)` | replaced `blockIdx.x` with `get_nd_item<3>().get_group(2)` |
| `tx→lid` | replaced `threadIdx.x/y` with `get_nd_item<3>().get_local_id(...)` |
| `syn→bar` | replaced `__syncthreads()` with SYCL barrier |
| `sh→loc` | replaced `__shared__` with `sycl::local` |
| `m2s→q` | replaced `cudaMemcpyToSymbol` with SYCL queue memcpy |
| `m2f→ptr=` | replaced `cudaMemcpyFromSymbol` with direct pointer assignment |
| `alloc` | `cudaMalloc`→`malloc_device` / `cudaMallocHost`→`malloc_host` |
| `dealloc` | `cudaFree`→`dpct_free` / `cudaFreeHost`→`sycl::free` |
| `fptr-memcpy` | function pointer transfers via SYCL queue memcpy |
| `q-memcpy` | `dpct::get_in_order_queue().memcpy().wait()` |
| `dpG` | `__CUDACC__` CPU/GPU dispatch guards preserved unchanged |
| `rn-guard` | renamed include guard |
| `renamed` | file renamed to neutral name |
| `cuFFTDx-TODO` | cuFFTDx SYCL-compatible FFT replacement needed (open TODO) |
| `full→SYCL` | fully migrated {component} to SYCL |
| `t2w` | "CUDA thread blocks"→"SYCL work-groups" in comments |
| `CUFLU→GPU` | renamed solver functions `CUFLU_`/`CUSRC_`/`CUPOT_`→`GPU_` |
| `cleanup` | cleaned up CUDA references in comments |
| `umul` | restored `__umul24`/`__mul24` via DPCT compat guard |

## Tracker entries

| Status | Original file | DPCT suggestion file | Notes |
| --- | --- | --- | --- |
| DONE | `src/Makefile_base` | None (no DPCT suggestion file) | `cleanup`: Removed all `.cu` compile rules (CUFLU_%.cu, CUPOT_%.cu, CUSRC_%.cu, %.cu) and NVCC-based GPU compile/link rules. GPU `.cpp` files now compiled with `$(CXX) $(CXXFLAG) $(SYCLFLAG)`. Added `SYCLFLAG = @@@SYCLFLAG@@@`. Updated GPU_FILE lists to use `.cpp` filenames (`.cu` files renamed or merged into CPU_*.cpp). Removed CUDA_PATH, CUFFTDX_PATH, NVCCFLAG_*, and `-lcudart` library linking. Removed GPU `-dlink` linker step. Removed `vpath %.cu` declarations. Fixed OBJ_GPU patsubst from `%.cu` to `%.cpp`. Updated TestProblem wildcard to filter GPU `.cpp` files (ExtAcc_*/ExtPot_*) from CPU files. |
| DONE | `src/configure.py` | None (no DPCT suggestion file) | Aligned with Makefile_base SYCL migration: removed NVCCFLAG_COM/FLU/POT from flags dict, removed NVCCFLAG_ARCH and MAXRREGCOUNT_FLU from set_gpu(), added SYCLFLAG (`-fsycl`); removed CUDA_PATH/CUFFTDX_PATH warnings; removed `--gpu_regcount_flu` argument; updated `--gpu` help text to reference SYCL. |
| DONE | `include/CheckError.h` | `dpct_out/include/CheckError.h` | `DEVICE_*` naming; neutral helper kept in main tree & `dpct_out/`. |
| DONE | `include/ConstMemory.h` | `dpct_out/include/ConstMemory.h` | `renamed` to neutral `ConstMemory.h`; use this name going forward. |
| DONE | `include/FLU.h` | `dpct_out/include/FLU.h` | `DPCT→SYCL` fluid-header migration; `C→S` in cuFFTDx section; replaced `cuFFTDx-TODO` — `#include "SyclFFT.h"` replaces cuFFTDx include & typedefs; guarded `__CUDACC_VER_MAJOR__` for CUDA-only path. |
| IN PROGRESS | `include/SyclFFT.h` | None (new file, see `doc/SyclFFT/DESIGN.md`) | New SYCL-native block FFT replacing cuFFTDx (mixed-radix 4/2/3/5/7 DIT, batched, in-place, unnormalized). Validated on host against a direct DFT for N = 64/72/108/168/300 (float & double, fwd & inv). Wired into `FLU.h`, `GPU_Asyn_FluidSolver.cpp`, and `CPU_ELBDMSolver_GramFE_FFT.cpp` (workspace create/FFT::size/size_of replacement, shared-mem fix, operator* removal); remaining: SYCL build/test. |
| DONE | `include/Global.h` | `dpct_out/include/Global.h` | SYCL-only guard rejecting `NCOMP_PASSIVE==0`; `C→S` in Grackle block. |
| DONE | `include/GPUAPI.h` | `dpct_out/include/GPUAPI.h` | `renamed` to neutral `GPUAPI.h`; all includes updated; keep neutral name. |
| DONE | `include/POT.h` | `dpct_out/include/POT.h` | `C→S`; device macros → `__dpct_inline__`/`__dpct_noinline__`; CGPU_LOOP updated (`tx→lid`, `.get_local_range(2)`, host+device); `umul`. |
| DONE | `src/EoS/Gamma/CPU_EoS_Gamma.cpp` | `dpct_out/src/EoS/Gamma/CPU_EoS_Gamma.cpp.dp.cpp` | `DPCT→SYCL`; `fptr-memcpy`; `C→S`; `cleanup`. |
| DONE | `src/EoS/GammaCR/CPU_EoS_GammaCR.cpp` | `dpct_out/src/EoS/GammaCR/CPU_EoS_GammaCR.cpp.dp.cpp` | `DPCT→SYCL`; `fptr-memcpy`; `C→S`; `cleanup`. |
| DONE | `src/EoS/Isothermal/CPU_EoS_Isothermal.cpp` | `dpct_out/src/EoS/Isothermal/CPU_EoS_Isothermal.cpp.dp.cpp` | `DPCT→SYCL`; `C→S` guards; `dpct::global_memory` storage; `fptr-memcpy` for EoS function pointers. |
| DONE | `src/EoS/TaubMathews/CPU_EoS_TaubMathews.cpp` | `dpct_out/src/EoS/TaubMathews/CPU_EoS_TaubMathews.cpp.dp.cpp` | `DPCT→SYCL`; `fptr-memcpy`; `C→S`; `cleanup`. |
| DONE | `src/EoS/User_Template/CPU_EoS_User_Template.cpp` | `dpct_out/src/EoS/User_Template/CPU_EoS_User_Template.cpp.dp.cpp` | `DPCT→SYCL`; `fptr-memcpy`; `C→S`; `cleanup`. |
| DONE | `src/GPU_API/GPU_Asyn_dtSolver.cpp` | `dpct_out/src/GPU_API/GPU_Asyn_dtSolver.dp.cpp` | `full→SYCL`: async dt solver wrappers & launch blocks via Stream submission & memcpy API. |
| DONE | `src/GPU/API/GPU_Asyn_FluidSolver.cpp` | `dpct_out/src/GPU/API/GPU_Asyn_FluidSolver.dp.cpp` | `full→SYCL`: async fluid solver wrappers & launch blocks via Stream submission & memcpy API; replaced cuFFTDx workspace creation (`cufftdx::make_workspace`) with empty `FFT::workspace_type()`/`IFFT::workspace_type()`; replaced `cufftdx::size_of<FFT>::value` → `FFT::size`; renamed `cufftdx_shared_memory_size` → `fft_shared_memory_size`. |
| DONE | `src/GPU_API/GPU_Asyn_PoissonGravitySolver.cpp` | None (no DPCT suggestion file) | `full→SYCL`: async `Stream[s]->memcpy()`; kernel launches → `submit()` + `cgh.parallel_for` + `sycl::nd_range<3>`; `dim3`→`dpct::dim3` (4 sites); all 5 `__global__` forward declarations (`GPU_PoissonSolver_SOR/MG`, `GPU_HydroGravitySolver`, `GPU_ELBDMGravitySolver`, `GPU_ELBDMGravitySolver_HamiltonJacobi`) wrapped with `SYCL_EXTERNAL`/`C→S` guards matching their migrated definitions; HYDRO gravity launch declares `sycl::local_accessor` for `s_pot_new`/`s_pot_old` (+`UNSPLIT_GRAVITY`) and passes raw ptrs to `GPU_HydroGravitySolver`, matching its migrated definition in `CPU_HydroGravitySolver.cpp`. `#else __global__` dead branches retained (cf. `GPU_ExtPotSolver` decl). NOT compile-verified (no dpcpp on node); full build additionally gated by `GPU_SetCache.cpp` (DONE) and SOR/MG kernel bodies (IN PROGRESS, L127/128). |
| DONE | `src/GPU_API/GPU_Asyn_SrcSolver.cpp` | `dpct_out/src/GPU_API/GPU_Asyn_SrcSolver.dp.cpp` | `full→SYCL`: async source solver wrappers & launch blocks via Stream submission & memcpy API. |
| DONE | `src/GPU_API/GPU_DiagnoseDevice.cpp` | `dpct_out/src/GPU_API/GPU_DiagnoseDevice.dp.cpp` | `full→SYCL`; mapped hardware logging (memory sizes, sub-group, WG sizes, CUs, ECC) to `dpct::device_info` & SYCL device props. |
| DONE | `src/GPU_API/GPU_MemAllocate_Fluid.cpp` | `dpct_out/src/GPU_API/GPU_MemAllocate_Fluid.dp.cpp` | `full→SYCL`; `alloc`; `dealloc`; queue destructions; `cleanup`. |
| DONE | `src/GPU_API/GPU_MemAllocate_PoissonGravity.cpp` | None (no DPCT suggestion file) | `full→SYCL`; `alloc` (`cudaMalloc`→`malloc_device`, `cudaMallocHost`→`malloc_host`) via `dpct::get_in_order_queue()`. |
| DONE | `src/GPU_API/GPU_MemAllocate.cpp` | `dpct_out/src/GPU_API/GPU_MemAllocate.dp.cpp` | Replaced legacy CUDA memory messages with vendor-neutral GPU messages; verified clean. |
| DONE | `src/GPU_API/GPU_MemFree_Fluid.cpp` | `dpct_out/src/GPU_API/GPU_MemFree_Fluid.dp.cpp` | `full→SYCL`; `dealloc`; `cleanup`. |
| DONE | `src/GPU_API/GPU_MemFree_PoissonGravity.cpp` | None (no DPCT suggestion file) | `full→SYCL`; `dealloc` (`cudaFree`→`dpct_free`, `cudaFreeHost`→`sycl::free`) via `dpct::get_in_order_queue()`. |
| DONE | `src/GPU_API/GPU_SendExtPotTable2GPU.cpp` | None (conditional compilation) | `full→SYCL`; `q-memcpy` for ExtPot table transfers. |
| DONE | `src/GPU_API/GPU_SendGramFEMatrix2GPU.cpp` | None (conditional compilation) | `full→SYCL`; `q-memcpy` for GramFE matrix transfers. |
| DONE | `src/GPU_API/GPU_SetCache.cpp` | `dpct_out/src/GPU_API/GPU_SetCache.dp.cpp` | `full→SYCL`; stubbed out legacy CUDA cache config APIs. All `__global__` forward declarations wrapped with `SYCL_EXTERNAL`/`C→S` guards matching migrated definitions: `GPU_HydroGravitySolver` updated with SYCL-only `s_pot_new`/`s_pot_old` (UNSPLIT_GRAVITY) params; `GPU_ELBDMGravitySolver_HamiltonJacobi` fwd decl added; `GPU_dtSolver_HydroCFL` updated with `real *shared` (SYCL); `GPU_FluidSolver_MHM/CTU` updated with `int *const c_NormIdx/c_FracIdx` (SYCL); `GPU_ELBDMSolver_GramFE_FFT` params fixed (`workspace`/`workspace_inverse`); `GPU_ELBDMSolver_GramFE_MATMUL` param fixed (`g_Evolve`); `GPU_ELBDMSolver_HamiltonJacobi` SYCL params use `h_` prefix. NOT compile-verified (no dpcpp on node). |
| DONE | `src/GPU_API/GPU_SetConstMemory_EoS.cpp` | `dpct_out/src/GPU_API/GPU_SetConstMemory_EoS.dp.cpp` | `full→SYCL`; EoS const-memory transfers & symbol addr resolution via `q-memcpy` + ptr assignments; removed `m2s` & `cudaGetSymbolAddress`. |
| DONE | `src/GPU_API/GPU_SetConstMemory_ExtAccPot.cpp` | `dpct_out/src/GPU_API/GPU_SetConstMemory_ExtAccPot.dp.cpp` | `full→SYCL`; ExtAcc/Pot const-memory transfers via `q-memcpy`; removed `m2s`. |
| DONE | `src/GPU_API/GPU_SetConstMemory.cpp` | `dpct_out/src/GPU_API/GPU_SetConstMemory.dp.cpp` | `full→SYCL`; const-memory transfers via `q-memcpy` across all preprocessor branches; removed `m2s`. |
| DONE | `src/GPU_API/GPU_SetDevice.cpp` | `dpct_out/src/GPU_API/GPU_SetDevice.dp.cpp` | `full→SYCL`; updated device count, selection, property queries (WG/sub-group sizes), warning checks, temp memory allocations. |
| DONE | `src/GPU_API/GPU_SetMemSize.cpp` | `dpct_out/src/GPU_API/GPU_SetMemSize.dp.cpp` | `full→SYCL`; multiProcessorCount→`get_max_compute_units()`; stubbed CUDA overlap attr query. |
| DONE | `src/GPU_API/GPU_Synchronize.cpp` | `dpct_out/src/GPU_API/GPU_Synchronize.dp.cpp` | `DPCT→SYCL` via `dpct::get_current_device().queues_wait_and_throw()`; `renamed` from `CUAPI_Synchronize.cu`; `cleanup`. |
| DONE | `src/GPU_Utility/BlockReduction_Shuffle.cpp` | `dpct_out/src/GPU_Utility/BlockReduction_Shuffle.dp.cpp` | Block reduction (shuffle) `→SYCL`; `+shared` param under `C→S`; `sh→loc` & `syn→bar` guarded CUDA-only; explicit SYCL barriers for SYCL path. |
| DONE | `src/GPU_Utility/BlockReduction_WarpSync.cpp` | None (no DPCT suggestion file) | Block reduction (warp-sync) `→SYCL`; `__clz`→`__builtin_clz` macro; `+shared` param under `C→S`; `sh→loc` & `syn→bar` guarded CUDA-only; `volatile`→`sycl::group_barrier` subgroup barriers for SYCL path. |
| DONE | `src/Microphysics/CosmicRayDiffusion/CPU_CR_AddDiffuseFlux.cpp` | None (no DPCT suggestion file) | Manual migration; `rn-guard` (`__CUFLU_*`→neutral); `+inc`; `C→S` (include + `__syncthreads` sites); `syn→bar` via `get_nd_item<3>().barrier()`; `CUFLU→GPU` in comments; updated `#else`/`#endif` comments. |
| IN PROGRESS | `src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_GramFE_FFT.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `GAMER→Macro`; `C→S`; `d→DEV`; `sh→loc`; `tx→lid`; `bx→grp(2)`; `syn→bar`; `g→EXT`; removed `__launch_bounds__`; fixed `cuFFTDx-TODO` — replaced ambiguous complex operator overloads (now uses std::complex); fixed shared-mem array size (was bytes, now elements: `ffts_per_block * (GRAMME_FLU_NXT + 2*GRAMME_NDELTA)`). |
| DONE | `src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_GramFE_MATMUL.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `GAMER→Macro`; `C→S`; `d→DEV`; `sh→loc`; `tx→lid`; `bx→grp(2)`; `syn→bar`; `g→EXT`; 'CUDA'→'device compiler' comment. |
| DONE | `src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_HJ.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `GAMER→Macro` (before `FLU.h`); `C→S`; `g→EXT`; `sh→loc` (CGPU_SHARED); `tx→lid`; `bx→grp(2)`; `syn→bar`; `t2w`. |
| DONE | `src/Model_ELBDM/CPU_ELBDMGravity/CPU_ELBDMGravitySolver_HJ.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `GAMER→Macro` + `POT.h`; `C→S`; `g→EXT`; `bx→grp(2)`; `t2w`. |
| DONE | `src/Model_ELBDM/CPU_ELBDMGravity/CPU_ELBDMGravitySolver.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `GAMER→Macro` + `POT.h`; `C→S`; `g→EXT`; `bx→grp(2)`; `t2w`. |
| DONE | `src/Model_ELBDM/GPU_ELBDM/GPU_ELBDMSolver_FD.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `full→SYCL`; `sh→loc` (CGPU_SHARED macro mapping to `sycl::local`); `syn→bar` (4 sites, guarded `#ifdef SYCL_LANGUAGE_VERSION`); `bx→grp(2)`/`tx→lid` (`blockIdx`/`threadIdx`→`get_nd_item<3>().get_group(2)`/`get_local_id(...)` guarded for CUDA fallback). `__umul24`/`__mul24` already covered by FLU.h compat macros. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_dtSolver_HydroCFL.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_dtSolver_HydroCFL.cpp.dp.cpp` | `DPCT→SYCL`; `+inc`; `C→S`; `g→EXT`; `bx→grp(2)`; `tx→lid` (`threadIdx.x==0`); `+shared` param (matching caller `GPU_Asyn_dtSolver.cpp`); updated `BlockReduction_*` calls; `t2w`. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_FluidSolver_CTU.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_FluidSolver_CTU.cpp.dp.cpp` | `DPCT→SYCL`; `g→EXT`; `C→S`; `bx→grp(2)`; `t2w`; `syn→bar` completed in prior pass. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_FluidSolver_MHM.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_FluidSolver_MHM.cpp.dp.cpp` | `DPCT→SYCL`; `g→EXT`; `C→S`; `bx→grp(2)`; `sh→loc` (`s_FullStepFailure` now `sycl::local int` in GPU path); `t2w`; `syn→bar` completed in prior pass. Resolved: `atomicExch`/`__CUDA_ARCH__` issue in `CPU_Shared_FullStepUpdate.cpp` (now `sycl::atomic_ref`). |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_ComputeFlux.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_ComputeFlux.cpp` | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`; `syn→bar`; DPCT1110 register pressure comment. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_ConstrainedTransport.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`; `syn→bar`. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_DataReconstruction.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_DataReconstruction.cpp` | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`; `syn→bar`. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_DualEnergy.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_FluUtility.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_FluUtility.cpp` | CUDA→SYCL: `rn-guard`; `C→S` (all preprocessor guards); `+inc`. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_FullStepUpdate.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_FullStepUpdate.cpp` | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`; `syn→bar`; `atomicExch`/`__CUDA_ARCH__`→`sycl::atomic_ref` (work_group scope, relaxed); `cleanup` (comments: "GPU thread block"→"SYCL work-group", fixed circular migration note). |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_Exact.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_Exact.cpp` | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_HLLC.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_HLLC.cpp` | `DPCT→SYCL` header incl. & guard switch; include now uses `FLU.h` name. `rn-guard` (`__CUFLU_*`→neutral); `C→S`; `+inc`; DPCT1110 register pressure comment. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_HLLD.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_HLLD.cpp` | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_HLLE.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_HLLE.cpp` | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`. |
| DONE | `src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_Roe.cpp` | `dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_Roe.cpp` | CUDA→SYCL: `rn-guard`; `C→S`; `+inc`. |
| IN PROGRESS | `src/Model_Hydro/GPU_Hydro/GPU_FluidSolver_RTVD.cpp` | `dpct_out/src/Model_Hydro/GPU_Hydro/GPU_FluidSolver_RTVD.dp.cpp` | `full→SYCL`: fluid solver kernel; `+hdrs`; `g→EXT` & `d→DEV`; preserved `CPU_Shared_FluUtility.cpp` helper include. BUT: kernel body `__shared__`/`__syncthreads()`/`__mul24`/`__umul24` remain un-guarded — need `sh→loc`+`syn→bar`. |
| DONE | `src/SelfGravity/CPU_Gravity/CPU_ExtAcc_PointMass.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtAcc_PointMass`). |
| DONE | `src/SelfGravity/CPU_Poisson/CPU_ExtPot_PointMass.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_PointMass`). |
| DONE | `src/SelfGravity/CPU_Poisson/CPU_ExtPot_Tabular.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_Tabular`). |
| DONE | `src/SelfGravity/CPU_Poisson/CPU_ExtPotSolver.cpp` | `dpct_out/src/SelfGravity/CPU_Poisson/CPU_ExtPotSolver.cpp.dp.cpp` | CUDA→SYCL (`GPU_ExtPotSolver`): `+hdrs`; `g→EXT` (parallel to CUDA `__global__`); `bx→grp(2)` (patch loop); `C→S` (ConstMemory.h guard); fwd decl in `GPU_Asyn_PoissonGravitySolver.cpp` updated. |
| IN PROGRESS | `src/SelfGravity/GPU_Poisson/GPU_PoissonSolver_MG.cpp` | None (no DPCT suggestion file) | `full→SYCL`: MG Poisson solver; `renamed` from `CUPOT_PoissonSolver_MG.cu`; `+hdrs`; `g→EXT` & `d→DEV`; `bx→grp(2)`/`tx→lid`; `__clz`→`__builtin_clz`; `Makefile`/`Makefile_base` updated. BUT: kernel body `__shared__`/`__syncthreads()`/`__mul24`/`__umul24` remain un-guarded — need `sh→loc`+`syn→bar`. |
| IN PROGRESS | `src/SelfGravity/GPU_Poisson/GPU_PoissonSolver_SOR.cpp` | None (no DPCT suggestion file) | `full→SYCL`: SOR Poisson solver; `+hdrs`; `renamed` to `.cpp`; `g→EXT`; `bx→grp(2)`/`tx→lid`/`blockDim`; `Makefile`/`Makefile_base` updated. BUT: kernel body `__shared__`/`__syncthreads()`/`__mul24`/`__umul24` remain un-guarded — need `sh→loc`+`syn→bar`. |
| DONE | `src/SourceTerms/CPU_SrcSolver_IterateAllCells.cpp` | `dpct_out/src/SourceTerms/CPU_SrcSolver_IterateAllCells.cpp.dp.cpp` | `DPCT→SYCL`; `+inc`; `C→S`; `g→EXT`; `bx→grp(2)`; `t2w`; omitted DPCT1102 diagnostic comment. |
| DONE | `src/SourceTerms/Deleptonization/CPU_Src_Deleptonization.cpp` | `dpct_out/src/SourceTerms/Deleptonization/CPU_Src_Deleptonization.cpp.dp.cpp` | `DPCT→SYCL`; `fptr-memcpy`, const-memory & profile transfers via `q-memcpy`; `C→S`; `cleanup`. |
| DONE | `src/SourceTerms/ExactCooling/CPU_Src_ExactCooling.cpp` | `dpct_out/src/SourceTerms/ExactCooling/CPU_Src_ExactCooling.cpp.dp.cpp` | `DPCT→SYCL`; `alloc` (`dpct::dpct_malloc`); `dealloc` (`dpct::dpct_free`); `fptr-memcpy`, const-memory & profile via `q-memcpy`; `C→S`; `cleanup`. |
| DONE | `src/SourceTerms/User_Template/CPU_Src_User_Template.cpp` | `dpct_out/src/SourceTerms/User_Template/CPU_Src_User_Template.cpp.dp.cpp` | `DPCT→SYCL`; `fptr-memcpy`, const-memory via `q-memcpy`; `C→S`; updated include (`CPU_Shared_FluUtility.cpp`). |
| DONE | `src/TestProblem/ELBDM/DiskHeating/ExtPot_Soliton.cpp` | `dpct_out/src/TestProblem/ELBDM/DiskHeating/ExtPot_Soliton.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_Soliton`); `dpG`. |
| DONE | `src/TestProblem/ELBDM/ExtPot/ExtPot_ELBDM_ExtPot.cpp` | `dpct_out/src/TestProblem/ELBDM/ExtPot/ExtPot_ELBDM_ExtPot.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_ELBDM_ExtPot`); `dpG`. |
| DONE | `src/TestProblem/ELBDM/HaloMerger/ExtPot_ELBDM_HaloMerger.cpp` | `dpct_out/src/TestProblem/ELBDM/HaloMerger/ExtPot_ELBDM_HaloMerger.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_ELBDM_HaloMerger`); `dpG`. |
| DONE | `src/TestProblem/Hydro/Bondi/ExtAcc_Bondi.cpp` | `dpct_out/src/TestProblem/Hydro/Bondi/ExtAcc_Bondi.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtAcc_Bondi`); `dpG`. |
| DONE | `src/TestProblem/Hydro/CMZ/CPU_ExtPot_TabularP17.cpp` | `dpct_out/src/TestProblem/Hydro/CMZ/CPU_ExtPot_TabularP17.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_TabularP17`); `dpG`. |
| DONE | `src/TestProblem/Hydro/CMZ/ExtAcc_LogBar.cpp` | `dpct_out/src/TestProblem/Hydro/CMZ/ExtAcc_LogBar.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtAcc_BarredPot`); `dpG`. |
| DONE | `src/TestProblem/Hydro/CMZ/ExtPot_LogBar.cpp` | `dpct_out/src/TestProblem/Hydro/CMZ/ExtPot_LogBar.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_BarredPot`); `dpG`. |
| DONE | `src/TestProblem/Hydro/Jet/ExtAcc_Jet.cpp` | `dpct_out/src/TestProblem/Hydro/Jet/ExtAcc_Jet.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtAcc_Jet`); `dpG`. |
| DONE | `src/TestProblem/Hydro/ParticleEquilibriumIC/ExtPot_ParEqmIC.cpp` | `dpct_out/src/TestProblem/Hydro/ParticleEquilibriumIC/ExtPot_ParEqmIC.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_ParEqmIC`); `dpG`. |
| DONE | `src/TestProblem/Hydro/Plummer/ExtAcc_Plummer.cpp` | `dpct_out/src/TestProblem/Hydro/Plummer/ExtAcc_Plummer.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtAcc_Plummer`); `dpG`. |
| DONE | `src/TestProblem/Hydro/Plummer/ExtPot_Plummer.cpp` | `dpct_out/src/TestProblem/Hydro/Plummer/ExtPot_Plummer.cpp.dp.cpp` | CUDA→SYCL: `+hdrs`; `C→S` (CheckError.h); `m2f→ptr=` (`SetGPUExtPot_Plummer`); `dpG`. |
| DONE | `test_problem_deprecated/Model_ParticleOnly/TwoParOrbit/GPU_ExternalAcc.1.1.0.cpp` | None (no DPCT suggestion file) | `renamed` from `CUPOT_ExternalAcc.1.1.0.cu`; `C→S` guards; `d→DEV`; `+hdrs`; updated `Copy_ExtAcc.sh` & `Init_ExternalAcc.cpp`. |
| DONE | `test_problem_deprecated/Model_ParticleOnly/TwoParOrbit/GPU_ExternalPot.1.0.0.cpp` | None (no DPCT suggestion file) | `renamed` from `CUPOT_ExternalPot.1.0.0.cu`; `C→S` guards; `d→DEV`; `+hdrs`; updated `Copy_ExtPot.sh` & `Init_ExternalPot.cpp`. |
| DONE | `src/Model_Hydro/CPU_HydroGravity/CPU_HydroGravitySolver.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `C→S` (ConstMemory.h + all dispatch guards `__CUDACC`→`SYCL_LANGUAGE_VERSION`); `g→EXT` (`__global__`→`SYCL_EXTERNAL void GPU_HydroGravitySolver`); `tx→lid`/`bx→grp(2)` (`threadIdx.x`→`get_local_id(2)`, `blockIdx.x`→`get_group(2)`); `sh→loc` (`__shared__` → `sycl::local_accessor` passed as `real *s_pot_new`/`s_pot_old` params from the `GPU_Asyn_PoissonGravitySolver.cpp` HYDRO launch site); `syn→bar` (`__syncthreads()`→`item_ct1.barrier(sycl::access::fence_space::local_space)`); CPU OpenMP path preserved. No residual CUDA-isms; signature matches the forward declaration in `GPU_Asyn_PoissonGravitySolver.cpp`. NOT compile-verified (no dpcpp on node). |
| DONE | `src/Model_Hydro/CPU_HydroGravity/CPU_dtSolver_HydroGravity.cpp` | None (no DPCT suggestion file) | CUDA→SYCL: `C→S` (`__CUDACC__`→`SYCL_LANGUAGE_VERSION` for GPU setup, kernel declaration, shared-mem load, patch loop, reduction); `g→EXT` (`__global__`→`SYCL_EXTERNAL`); `sh→loc` (`__shared__ real s_Pot`→`sycl::local real s_Pot`); `syn→bar` (`__syncthreads()`→`item_ct1.barrier(...)`); `tx→lid` (`threadIdx.x`→`get_local_id(2)`); `bx→grp(2)` (`blockIdx.x`→`get_group(2)`); `+shared` (added `real *shared` param for BlockReduction, allocated via `sycl::local_accessor` in dispatch); `t2w`. CPU OpenMP path preserved. Forward declarations in `GPU_Asyn_dtSolver.cpp` and `GPU_SetCache.cpp` updated to match. NOT compile-verified (no dpcpp on node). |

