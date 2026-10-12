#ifndef __SRC_DISPATCH_H__
#define __SRC_DISPATCH_H__


// ****************************************************************************
// ** This header provides compile-time source-term dispatch for SYCL device  **
// ** code. SYCL kernels cannot call through function pointers. Since the     **
// ** concrete source-term function name is fixed at compile time (e.g.,     **
// ** Src_Deleptonization, Src_ExactCooling, Src_User_Template), we replace   **
// ** all runtime function-pointer indirection with direct calls under SYCL.  **
// ** This mirrors the EoS_Dispatch.h pattern already used for the EoS        **
// ** functions.                                                               **
// ****************************************************************************


#include "SrcTerms.h"  // for SrcTerms_t (also brings in EoS.h / EoS_t)


// ---------------------------------------------------------------------------
// Macro: SrcExternDecl
//   Under SYCL: expands to SYCL_EXTERNAL so that device functions defined in
//               one TU (e.g., CPU_Src_Deleptonization.cpp) can be linked from
//               device code in another TU (e.g., CPU_SrcSolver_IterateAllCells.cpp).
//   Under CPU:  empty (regular extern linkage is sufficient).
// ---------------------------------------------------------------------------
#ifdef SYCL_LANGUAGE_VERSION
#  define SrcExternDecl SYCL_EXTERNAL
#else
#  define SrcExternDecl
#endif


// ===========================================================================
// SYCL-only: extern declarations for concrete source-term functions
// ===========================================================================
#ifdef SYCL_LANGUAGE_VERSION


#if ( MODEL == HYDRO )
SrcExternDecl void Src_Deleptonization( real fluid[], const real B[],
   const SrcTerms_t *SrcTerms, const real dt, const real dh,
   const double x, const double y, const double z,
   const double TimeNew, const double TimeOld,
   const real MinDens, const real MinPres, const real MinEint, const long PassiveFloor,
   const EoS_t *EoS, const double AuxArray_Flt[], const int AuxArray_Int[] );
#endif


#ifdef EXACT_COOLING
SrcExternDecl void Src_ExactCooling( real fluid[], const real B[], const SrcTerms_t *SrcTerms, const real dt,
   const real dh, const double x, const double y, const double z,
   const double TimeNew, const double TimeOld,
   const real MinDens, const real MinPres, const real MinEint, const long PassiveFloor,
   const EoS_t *EoS, const double AuxArray_Flt[], const int AuxArray_Int[] );
#endif


// User-specified source term — the function name is always Src_User_Template
// by convention (users edit the template file in place, not rename functions).
SrcExternDecl void Src_User_Template( real fluid[], const real B[],
   const SrcTerms_t *SrcTerms, const real dt, const real dh,
   const double x, const double y, const double z,
   const double TimeNew, const double TimeOld,
   const real MinDens, const real MinPres, const real MinEint, const long PassiveFloor,
   const EoS_t *EoS, const double AuxArray_Flt[], const int AuxArray_Int[] );


#endif // #ifdef SYCL_LANGUAGE_VERSION


// ===========================================================================
// CALL_SRC_* macros
//
// These macros allow the same source code to work in both SYCL and non-SYCL
// paths:
//   - Under SYCL_LANGUAGE_VERSION: call the concrete source-term function
//     directly (compile-time dispatch; SYCL kernels cannot call through
//     function pointers).
//   - Under non-SYCL (CPU): call through the function pointer stored in the
//     SrcTerms_t struct member.
//
// The third named parameter (pSrcTerms) is a pointer to the SrcTerms_t
// struct. Under SYCL it is forwarded as the SrcTerms_t* argument to the
// concrete function; under CPU it is used to dereference the function pointer.
//
// Usage: replace SrcTerms.Dlep_FuncPtr(fluid, B, &SrcTerms, ...) with
//        CALL_SRC_Deleptonization(fluid, B, &SrcTerms, ...)
//        and similarly for ExactCooling and User.
// ===========================================================================
#ifdef SYCL_LANGUAGE_VERSION

#  if ( MODEL == HYDRO )
#    define CALL_SRC_Deleptonization(fluid, B, pSrcTerms, ...) \
           Src_Deleptonization(fluid, B, pSrcTerms, __VA_ARGS__)
#  endif

#  ifdef EXACT_COOLING
#    define CALL_SRC_ExactCooling(fluid, B, pSrcTerms, ...) \
           Src_ExactCooling(fluid, B, pSrcTerms, __VA_ARGS__)
#  endif

#    define CALL_SRC_User(fluid, B, pSrcTerms, ...) \
           Src_User_Template(fluid, B, pSrcTerms, __VA_ARGS__)

#else

#  if ( MODEL == HYDRO )
#    define CALL_SRC_Deleptonization(fluid, B, pSrcTerms, ...) \
           pSrcTerms->Dlep_FuncPtr(fluid, B, pSrcTerms, __VA_ARGS__)
#  endif

#  ifdef EXACT_COOLING
#    define CALL_SRC_ExactCooling(fluid, B, pSrcTerms, ...) \
           pSrcTerms->EC_FuncPtr(fluid, B, pSrcTerms, __VA_ARGS__)
#  endif

#    define CALL_SRC_User(fluid, B, pSrcTerms, ...) \
           pSrcTerms->User_FuncPtr(fluid, B, pSrcTerms, __VA_ARGS__)

#endif


#endif // #ifndef __SRC_DISPATCH_H__
