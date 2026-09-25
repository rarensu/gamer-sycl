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
| IN PROGRESS | `/home/rlawrence/Projects/gamer-sycl/include/FLU.h` | `/home/rlawrence/Projects/gamer-sycl/dpct_out/include/CUFLU.h` | Applied the DPCT fluid-header migration, but the cuFFTDx-dependent GramFE FFT block still needs SYCL-side replacement. |
| DONE | `/home/rlawrence/Projects/gamer-sycl/include/CheckError.h` | `/home/rlawrence/Projects/gamer-sycl/dpct_out/include/CUDA_CheckError.h` | Renamed the CUDA-prefixed helper identifiers to `DEVICE_*` and updated all consumers in both the main tree and `dpct_out/` to match. |
