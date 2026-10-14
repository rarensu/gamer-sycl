#ifndef __EOS_DISPATCH_H__
#define __EOS_DISPATCH_H__


// ****************************************************************************
// ** This header provides compile-time EoS dispatch for SYCL device code. **
// ** SYCL kernels cannot call through function pointers. Since the EoS     **
// ** type (EOS macro) is a compile-time constant, we replace all        **
// ** runtime function-pointer indirection with compile-time dispatch    **
// ** via preprocessor #if chains.                                          **
// ****************************************************************************


// ---------------------------------------------------------------------------
// Macro: EoSExternDecl
//   Under SYCL: expands to SYCL_EXTERNAL so that device functions defined in
//               one TU (e.g., CPU_EoS_Gamma.cpp) can be linked from device
//               code in another TU (e.g., the solver files that #include
//               the shared sources).
//   Under CPU:  static (internal linkage). These "shared" EoS files are
//               compiled twice — once as a GPU/SYCL TU and once as a CPU TU.
//               Making CPU-side functions static (file-scoped) avoids
//               multiple-definition linker errors while still allowing the
//               static function-pointer initialization to take the function
//               address (resolved from the local TU, not the GPU object).
// ---------------------------------------------------------------------------
#ifndef EoSExternDecl
#  ifdef SYCL_LANGUAGE_VERSION
#    define EoSExternDecl SYCL_EXTERNAL
#  else
#    define EoSExternDecl static
#  endif
#endif


// ==========================================================================
// SYCL-only: extern declarations for concrete EoS functions + dispatch wrappers
// ==========================================================================
#ifdef SYCL_LANGUAGE_VERSION
#include <cmath>  // for NAN


// ---------- Extern declarations for concrete EoS functions ----------
#if   ( EOS == EOS_GAMMA )
EoSExternDecl real EoS_DensEint2Pres_Gamma( const real Dens, const real Eint, const real Passive[],
                                            const double AuxArray_Flt[], const int AuxArray_Int[],
                                            const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2Eint_Gamma( const real Dens, const real Pres, const real Passive[],
                                            const double AuxArray_Flt[], const int AuxArray_Int[],
                                            const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2CSqr_Gamma( const real Dens, const real Pres, const real Passive[],
                                            const double AuxArray_Flt[], const int AuxArray_Int[],
                                            const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensEint2Temp_Gamma( const real Dens, const real Eint, const real Passive[],
                                            const double AuxArray_Flt[], const int AuxArray_Int[],
                                            const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensTemp2Pres_Gamma( const real Dens, const real Temp, const real Passive[],
                                            const double AuxArray_Flt[], const int AuxArray_Int[],
                                            const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensEint2Entr_Gamma( const real Dens, const real Eint, const real Passive[],
                                            const double AuxArray_Flt[], const int AuxArray_Int[],
                                            const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl void EoS_General_Gamma( const int Mode, real Out[], const real In_Flt[], const int In_Int[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX] );

#elif ( EOS == EOS_ISOTHERMAL )
EoSExternDecl real EoS_DensEint2Pres_Isothermal( const real Dens, const real Eint, const real Passive[],
                                                 const double AuxArray_Flt[], const int AuxArray_Int[],
                                                 const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2Eint_Isothermal( const real Dens, const real Pres, const real Passive[],
                                                 const double AuxArray_Flt[], const int AuxArray_Int[],
                                                 const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2CSqr_Isothermal( const real Dens, const real Pres, const real Passive[],
                                                 const double AuxArray_Flt[], const int AuxArray_Int[],
                                                 const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensEint2Temp_Isothermal( const real Dens, const real Eint, const real Passive[],
                                                 const double AuxArray_Flt[], const int AuxArray_Int[],
                                                 const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensTemp2Pres_Isothermal( const real Dens, const real Temp, const real Passive[],
                                                 const double AuxArray_Flt[], const int AuxArray_Int[],
                                                 const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensEint2Entr_Isothermal( const real Dens, const real Eint, const real Passive[],
                                                 const double AuxArray_Flt[], const int AuxArray_Int[],
                                                 const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl void EoS_General_Isothermal( const int Mode, real Out[], const real In_Flt[], const int In_Int[],
                                            const double AuxArray_Flt[], const int AuxArray_Int[],
                                            const real *const Table[EOS_NTABLE_MAX] );

#elif ( EOS == EOS_TAUBMATHEWS )
EoSExternDecl real EoS_GuessHTilde_TaubMathews( const real Con[], real* const Constant,
                                                const double AuxArray_Flt[], const int AuxArray_Int[],
                                                const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl void EoS_HTilde2Temp_TaubMathews( const real HTilde, real* const Temp, real* const DiffTemp,
                                                const real Passive[], const double AuxArray_Flt[],
                                                const int AuxArray_Int[], const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_Temp2HTilde_TaubMathews( const real Temp, const real Passive[],
                                                const double AuxArray_Flt[], const int AuxArray_Int[],
                                                const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2CSqr_TaubMathews( const real Dens, const real Pres, const real Passive[],
                                                  const double AuxArray_Flt[], const int AuxArray_Int[],
                                                  const real *const Table[EOS_NTABLE_MAX] );


#elif ( EOS == EOS_COSMIC_RAY )
EoSExternDecl real EoS_DensEint2Pres_GammaCR( const real Dens, const real Eint, const real Passive[],
                                              const double AuxArray_Flt[], const int AuxArray_Int[],
                                              const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2Eint_GammaCR( const real Dens, const real Pres, const real Passive[],
                                              const double AuxArray_Flt[], const int AuxArray_Int[],
                                              const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2CSqr_GammaCR( const real Dens, const real Pres, const real Passive[],
                                              const double AuxArray_Flt[], const int AuxArray_Int[],
                                              const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensEint2Temp_GammaCR( const real Dens, const real Eint, const real Passive[],
                                              const double AuxArray_Flt[], const int AuxArray_Int[],
                                              const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensTemp2Pres_GammaCR( const real Dens, const real Temp, const real Passive[],
                                              const double AuxArray_Flt[], const int AuxArray_Int[],
                                              const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensEint2Entr_GammaCR( const real Dens, const real Eint, const real Passive[],
                                              const double AuxArray_Flt[], const int AuxArray_Int[],
                                              const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl void EoS_General_GammaCR( const int Mode, real Out[], const real In_Flt[], const int In_Int[],
                                         const double AuxArray_Flt[], const int AuxArray_Int[],
                                         const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_CREint2CRPres_GammaCR( const real E_CR, const double AuxArray_Flt[],
                                              const int AuxArray_Int[], const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_GasPres2GasEint_GammaCR( const real Pres_Gas, const double AuxArray_Flt[],
                                                const int AuxArray_Int[], const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_GasEint2GasPres_GammaCR( const real Eint_Gas, const double AuxArray_Flt[],
                                                const int AuxArray_Int[], const real *const Table[EOS_NTABLE_MAX] );

#elif ( EOS == EOS_USER )
EoSExternDecl real EoS_DensEint2Pres_User_Template( const real Dens, const real Eint, const real Passive[],
                                                    const double AuxArray_Flt[], const int AuxArray_Int[],
                                                    const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2Eint_User_Template( const real Dens, const real Pres, const real Passive[],
                                                    const double AuxArray_Flt[], const int AuxArray_Int[],
                                                    const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensPres2CSqr_User_Template( const real Dens, const real Pres, const real Passive[],
                                                    const double AuxArray_Flt[], const int AuxArray_Int[],
                                                    const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensEint2Temp_User_Template( const real Dens, const real Eint, const real Passive[],
                                                    const double AuxArray_Flt[], const int AuxArray_Int[],
                                                    const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensTemp2Pres_User_Template( const real Dens, const real Temp, const real Passive[],
                                                    const double AuxArray_Flt[], const int AuxArray_Int[],
                                                    const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl real EoS_DensEint2Entr_User_Template( const real Dens, const real Eint, const real Passive[],
                                                    const double AuxArray_Flt[], const int AuxArray_Int[],
                                                    const real *const Table[EOS_NTABLE_MAX] );
EoSExternDecl void EoS_General_User_Template( const int Mode, real Out[], const real In_Flt[], const int In_Int[],
                                               const double AuxArray_Flt[], const int AuxArray_Int[],
                                               const real *const Table[EOS_NTABLE_MAX] );

#elif !defined(EOS)
// EOS is not defined - this is allowed for models like ELBDM that don't use
// an equation of state. The dispatch wrappers below will return NAN if called.
#else
#  error : EoS_Dispatch.h: Unsupported EOS value. Supported values are: EOS_GAMMA, EOS_ISOTHERMAL, EOS_NUCLEAR, EOS_TABULAR, EOS_COSMIC_RAY, EOS_TAUBMATHEWS, EOS_USER
#endif


// ---------- Inline dispatch wrappers ----------
// Each wrapper selects the concrete EoS function at compile time.
//
// Fallback for unimplemented EoS functions:
// Some EoS types (notably EOS_TAUBMATHEWS) do not implement all standard
// conversion functions. Under the original CUDA runtime-dispatch model, these
// function pointers were simply never set and never called in the SRHD path.
// SYCL cannot call through function pointers in kernels, so these wrappers
// provide compile-time dispatch. When an EoS type omits a particular function,
// the fallback returns NAN at runtime instead of using #error — this mirrors
// the original CUDA behavior where calling an unset pointer would be a runtime
// error (segfault), not a compile error. NAN propagation triggers GAMER's
// auto-correction (OPT__1ST_FLUX_CORR / AUTO_REDUCE_DT).

static inline real GPU_EoS_DensEint2Pres( const real Dens, const real Eint, const real Passive[],
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
#  if   ( EOS == EOS_GAMMA )
    return EoS_DensEint2Pres_Gamma( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_ISOTHERMAL )
    return EoS_DensEint2Pres_Isothermal( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_COSMIC_RAY )
    return EoS_DensEint2Pres_GammaCR( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_USER )
    return EoS_DensEint2Pres_User_Template( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  else
#    ifdef GAMER_DEBUG
   printf( "ERROR : GPU_EoS_DensEint2Pres not supported for this EoS.\n" );
#    endif
   return NAN;
#  endif
}

static inline real GPU_EoS_DensPres2Eint( const real Dens, const real Pres, const real Passive[],
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
#  if   ( EOS == EOS_GAMMA )
    return EoS_DensPres2Eint_Gamma( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_ISOTHERMAL )
    return EoS_DensPres2Eint_Isothermal( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_COSMIC_RAY )
    return EoS_DensPres2Eint_GammaCR( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_USER )
    return EoS_DensPres2Eint_User_Template( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  else
#    ifdef GAMER_DEBUG
   printf( "ERROR : GPU_EoS_DensPres2Eint not supported for this EoS.\n" );
#    endif
   return NAN;
#  endif
}

static inline real GPU_EoS_DensPres2CSqr( const real Dens, const real Pres, const real Passive[],
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
#  if   ( EOS == EOS_GAMMA )
    return EoS_DensPres2CSqr_Gamma( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_ISOTHERMAL )
    return EoS_DensPres2CSqr_Isothermal( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_COSMIC_RAY )
    return EoS_DensPres2CSqr_GammaCR( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_TAUBMATHEWS )
    return EoS_DensPres2CSqr_TaubMathews( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_USER )
    return EoS_DensPres2CSqr_User_Template( Dens, Pres, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  else
#    ifdef GAMER_DEBUG
   printf( "ERROR : GPU_EoS_DensPres2CSqr not supported for this EoS.\n" );
#    endif
   return NAN;
#  endif
}

static inline real GPU_EoS_DensEint2Temp( const real Dens, const real Eint, const real Passive[],
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
#  if   ( EOS == EOS_GAMMA )
    return EoS_DensEint2Temp_Gamma( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_ISOTHERMAL )
    return EoS_DensEint2Temp_Isothermal( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_COSMIC_RAY )
    return EoS_DensEint2Temp_GammaCR( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_USER )
    return EoS_DensEint2Temp_User_Template( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  else
#    ifdef GAMER_DEBUG
   printf( "ERROR : GPU_EoS_DensEint2Temp not supported for this EoS.\n" );
#    endif
   return NAN;
#  endif
}

static inline real GPU_EoS_DensTemp2Pres( const real Dens, const real Temp, const real Passive[],
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
#  if   ( EOS == EOS_GAMMA )
    return EoS_DensTemp2Pres_Gamma( Dens, Temp, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_ISOTHERMAL )
    return EoS_DensTemp2Pres_Isothermal( Dens, Temp, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_COSMIC_RAY )
    return EoS_DensTemp2Pres_GammaCR( Dens, Temp, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_USER )
    return EoS_DensTemp2Pres_User_Template( Dens, Temp, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  else
#    ifdef GAMER_DEBUG
   printf( "ERROR : GPU_EoS_DensTemp2Pres not supported for this EoS.\n" );
#    endif
   return NAN;
#  endif
}

static inline real GPU_EoS_DensEint2Entr( const real Dens, const real Eint, const real Passive[],
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
#  if   ( EOS == EOS_GAMMA )
    return EoS_DensEint2Entr_Gamma( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_ISOTHERMAL )
    return EoS_DensEint2Entr_Isothermal( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_COSMIC_RAY )
    return EoS_DensEint2Entr_GammaCR( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  elif ( EOS == EOS_USER )
    return EoS_DensEint2Entr_User_Template( Dens, Eint, Passive, AuxArray_Flt, AuxArray_Int, Table );
#  else
#    ifdef GAMER_DEBUG
   printf( "ERROR : GPU_EoS_DensEint2Entr not supported for this EoS.\n" );
#    endif
   return NAN;
#  endif
}

#if defined( SRHD ) && ( EOS == EOS_TAUBMATHEWS )
static inline real GPU_EoS_GuessHTilde( const real Con[], real* const Constant,
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
    return EoS_GuessHTilde_TaubMathews( Con, Constant, AuxArray_Flt, AuxArray_Int, Table );
}

static inline void GPU_EoS_HTilde2Temp( const real HTilde, real* const Temp, real* const DiffTemp,
    const real Passive[], const double AuxArray_Flt[],
    const int AuxArray_Int[], const real *const Table[EOS_NTABLE_MAX] ) {
    EoS_HTilde2Temp_TaubMathews( HTilde, Temp, DiffTemp, Passive, AuxArray_Flt, AuxArray_Int, Table );
}

static inline real GPU_EoS_Temp2HTilde( const real Temp, const real Passive[],
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
    return EoS_Temp2HTilde_TaubMathews( Temp, Passive, AuxArray_Flt, AuxArray_Int, Table );
}
#endif // SRHD && EOS_TAUBMATHEWS

#if defined( COSMIC_RAY ) && ( EOS == EOS_COSMIC_RAY )
static inline real GPU_EoS_CREint2CRPres( const real E_CR,
    const double AuxArray_Flt[], const int AuxArray_Int[],
    const real *const Table[EOS_NTABLE_MAX] ) {
    return EoS_CREint2CRPres_GammaCR( E_CR, AuxArray_Flt, AuxArray_Int, Table );
}
#endif // COSMIC_RAY && EOS_COSMIC_RAY

#endif // SYCL_LANGUAGE_VERSION


// ==========================================================================
// CALL_EoS_* macros
//
// These macros allow the same source code to work in both SYCL and non-SYCL
// paths:
//   - Under SYCL_LANGUAGE_VERSION: call the compile-time dispatch wrapper GPU_EoS_*
//   - Under non-SYCL (CPU): call through the function pointer parameter directly
//
// Usage: replace EoS_DensEint2Pres(...) with CALL_EoS_DensEint2Pres(...)
// ==========================================================================
#ifdef SYCL_LANGUAGE_VERSION
#  define CALL_EoS_DensEint2Pres(...)    GPU_EoS_DensEint2Pres(__VA_ARGS__)
#  define CALL_EoS_DensPres2Eint(...)    GPU_EoS_DensPres2Eint(__VA_ARGS__)
#  define CALL_EoS_DensPres2CSqr(...)    GPU_EoS_DensPres2CSqr(__VA_ARGS__)
#  define CALL_EoS_DensEint2Temp(...)    GPU_EoS_DensEint2Temp(__VA_ARGS__)
#  define CALL_EoS_DensTemp2Pres(...)    GPU_EoS_DensTemp2Pres(__VA_ARGS__)
#  define CALL_EoS_DensEint2Entr(...)    GPU_EoS_DensEint2Entr(__VA_ARGS__)
#  if defined( SRHD ) && ( EOS == EOS_TAUBMATHEWS )
#    define CALL_EoS_GuessHTilde(...)    GPU_EoS_GuessHTilde(__VA_ARGS__)
#    define CALL_EoS_HTilde2Temp(...)    GPU_EoS_HTilde2Temp(__VA_ARGS__)
#    define CALL_EoS_Temp2HTilde(...)    GPU_EoS_Temp2HTilde(__VA_ARGS__)
#  endif
#  if defined( COSMIC_RAY ) && ( EOS == EOS_COSMIC_RAY )
#    define CALL_EoS_CREint2CRPres(...)  GPU_EoS_CREint2CRPres(__VA_ARGS__)
#  endif
#else
#  define CALL_EoS_DensEint2Pres(...)    EoS_DensEint2Pres(__VA_ARGS__)
#  define CALL_EoS_DensPres2Eint(...)    EoS_DensPres2Eint(__VA_ARGS__)
#  define CALL_EoS_DensPres2CSqr(...)    EoS_DensPres2CSqr(__VA_ARGS__)
#  define CALL_EoS_DensEint2Temp(...)    EoS_DensEint2Temp(__VA_ARGS__)
#  define CALL_EoS_DensTemp2Pres(...)    EoS_DensTemp2Pres(__VA_ARGS__)
#  define CALL_EoS_DensEint2Entr(...)    EoS_DensEint2Entr(__VA_ARGS__)
#  if defined( SRHD ) && ( EOS == EOS_TAUBMATHEWS )
#    define CALL_EoS_GuessHTilde(...)    EoS_GuessHTilde(__VA_ARGS__)
#    define CALL_EoS_HTilde2Temp(...)    EoS_HTilde2Temp(__VA_ARGS__)
#    define CALL_EoS_Temp2HTilde(...)    EoS_Temp2HTilde(__VA_ARGS__)
#  endif
#  if defined( COSMIC_RAY ) && ( EOS == EOS_COSMIC_RAY )
#    define CALL_EoS_CREint2CRPres(...)  EoS_CREint2CRPres(__VA_ARGS__)
#  endif
#endif


#endif // #ifndef __EOS_DISPATCH_H__
