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

## Suggested status format
- `TODO` — not yet applied to the original file
- `IN PROGRESS` — partially applied or under manual review
- `DONE` — DPCT suggestion has been applied to the original file and verified

## Tracker entries

| Status | Original file | DPCT suggestion file | Notes |
| --- | --- | --- | --- |
| DONE | `/home/rlawrence/Projects/gamer-sycl/src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_HLLC.cpp` | `/home/rlawrence/Projects/gamer-sycl/dpct_out/src/Model_Hydro/CPU_Hydro/CPU_Shared_RiemannSolver_HLLC.cpp` | Applied DPCT header/include migration and SYCL include guard switch; include now uses planned `FLU.h` name. |
| IN PROGRESS | `/home/rlawrence/Projects/gamer-sycl/include/FLU.h` | `/home/rlawrence/Projects/gamer-sycl/dpct_out/include/FLU.h` | Applied the DPCT fluid-header migration, but the cuFFTDx-dependent GramFE FFT block still needs SYCL-side replacement. |
| DONE | `/home/rlawrence/Projects/gamer-sycl/include/GPUAPI.h` | `/home/rlawrence/Projects/gamer-sycl/dpct_out/include/GPUAPI.h` | Neutral GPU API header is in place and all include statements now point to `GPUAPI.h`; keep using the neutral name rather than reviving legacy CUDA-prefixed variants. |
| DONE | `/home/rlawrence/Projects/gamer-sycl/include/CheckError.h` | `/home/rlawrence/Projects/gamer-sycl/dpct_out/include/CheckError.h` | Helper macros and functions now use the `DEVICE_*` names; keep the neutral helper naming in both the main tree and `dpct_out/`. |
| DONE | `/home/rlawrence/Projects/gamer-sycl/include/ConstMemory.h` | `/home/rlawrence/Projects/gamer-sycl/dpct_out/include/ConstMemory.h` | Constant-memory header has been renamed to the neutral `ConstMemory.h` and the tracker should use this filename going forward. |
| IN PROGRESS | `/home/rlawrence/Projects/gamer-sycl/include/POT.h` | `/home/rlawrence/Projects/gamer-sycl/dpct_out/include/POT.h` | Replaced `__CUDACC__` with `SYCL_LANGUAGE_VERSION` in include guards and GPU device specifiers; updated GPU device macros to `__dpct_inline__` and `__dpct_noinline__`; updated CGPU_LOOP to use SYCL `get_local_id(0)` and `get_local_size(0)`; commented experimental `__umul24` optimization pending SYCL equivalent. |
