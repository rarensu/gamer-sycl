#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUFLU.h"
#ifdef SYCL_LANGUAGE_VERSION
#include "CUDA_CheckError.h"
#include "CPU_Shared_FluUtility.cpp"
#endif

#if ( MODEL == HYDRO )



/********************************************************
1. Isothermal EoS (EOS_ISOTHERMAL)

2. This file is shared by both CPU and GPU

   GPU_EoS_Isothermal.cu -> CPU_EoS_Isothermal.cpp

3. Three steps are required to implement an EoS

   I.   Set EoS auxiliary arrays
   II.  Implement EoS conversion functions
   III. Set EoS initialization functions
********************************************************/



// =============================================
// I. Set EoS auxiliary arrays
// =============================================

//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_SetAuxArray_Isothermal
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
//                   AuxArray_Flt[0] = sound_speed^2
//                   AuxArray_Flt[1] = temperature in K
//
// Note        :  1. Invoked by EoS_Init_Isothermal()
//                2. AuxArray_Flt/Int[] have the size of EOS_NAUX_MAX defined in Macro.h (default = 20)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//                4. Physical constants such as Const_amu/Const_kB should be set to unity when disabling OPT__UNIT
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef SYCL_LANGUAGE_VERSION
void EoS_SetAuxArray_Isothermal( double AuxArray_Flt[], int AuxArray_Int[] )
{

// Cs^2 = kB*T/m = P/rho
   AuxArray_Flt[0] = ( OPT__UNIT ) ? ( Const_kB*ISO_TEMP/UNIT_E ) / ( MOLECULAR_WEIGHT*MU_NORM/UNIT_M )
                                   : ISO_TEMP / MOLECULAR_WEIGHT;
   AuxArray_Flt[1] = ISO_TEMP;

   if ( MPI_Rank == 0 )
   {
      Aux_Message( stdout, "   Temperature           = %13.7e K\n",    ISO_TEMP );
      Aux_Message( stdout, "   Mean molecular weight = %13.7e\n",      MOLECULAR_WEIGHT );
      if ( OPT__UNIT )
      Aux_Message( stdout, "   Sound speed           = %13.7e km/s\n", SQRT(AuxArray_Flt[0])*UNIT_V/Const_km );
      else
      Aux_Message( stdout, "   Sound speed           = %13.7e\n",      SQRT(AuxArray_Flt[0]) );
   }

#  ifdef GAMER_DEBUG
   real Cs2 = (real)AuxArray_Flt[0];
   Hydro_IsUnphysical_Single( Cs2, "sound speed squared", TINY_NUMBER, HUGE_NUMBER, ERROR_INFO, UNPHY_VERBOSE );
#  endif

} // FUNCTION : EoS_SetAuxArray_Isothermal
#endif // #ifndef __CUDACC__



// =============================================
// II. Implement EoS conversion functions
//     (1) EoS_DensEint2Pres_*
//     (2) EoS_DensPres2Eint_*
//     (3) EoS_DensPres2CSqr_*
//     (4) EoS_DensEint2Temp_* [OPTIONAL]
//     (5) EoS_DensTemp2Pres_* [OPTIONAL]
//     (6) EoS_DensEint2Entr_* [OPTIONAL]
//     (7) EoS_General_*       [OPTIONAL]
// =============================================

//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensEint2Pres_Isothermal
// Description :  Convert gas mass density and internal energy density to gas pressure
//
// Note        :  1. Internal energy density here is per unit volume instead of per unit mass
//                2. See EoS_SetAuxArray_Isothermal() for the values stored in AuxArray_Flt/Int[]
//
// Parameter   :  Dens       : Gas mass density
//                Eint       : Gas internal energy density
//                Passive    : Passive scalars (must not used here)
//                AuxArray_* : Auxiliary arrays (see the Note above)
//                Table      : EoS tables
//
// Return      :  Gas pressure
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensEint2Pres_Isothermal( const real Dens, const real Eint, const real Passive[],
                                          const double AuxArray_Flt[], const int AuxArray_Int[],
                                          const real *const Table[EOS_NTABLE_MAX] )
{

// check
#  ifdef GAMER_DEBUG
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );

   Hydro_IsUnphysical_Single( Dens, "input density", TINY_NUMBER, HUGE_NUMBER, ERROR_INFO, UNPHY_VERBOSE );
#  endif


   const real Cs2  = AuxArray_Flt[0];
   const real Pres = Cs2*Dens;

   return Pres;

} // FUNCTION : EoS_DensEint2Pres_Isothermal



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensPres2Eint_Isothermal
// Description :  Convert gas mass density and pressure to gas internal energy density
//
// Note        :  1. See EoS_DensEint2Pres_Isothermal()
//
// Parameter   :  Dens       : Gas mass density
//                Pres       : Gas pressure
//                Passive    : Passive scalars (must not used here)
//                AuxArray_* : Auxiliary arrays (see the Note above)
//                Table      : EoS tables
//
// Return      :  Gas internal energy density
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensPres2Eint_Isothermal( const real Dens, const real Pres, const real Passive[],
                                          const double AuxArray_Flt[], const int AuxArray_Int[],
                                          const real *const Table[EOS_NTABLE_MAX] )
{

// check
#  ifdef GAMER_DEBUG
   Hydro_IsUnphysical_Single( Pres, "input pressure", (real)0.0, HUGE_NUMBER, ERROR_INFO, UNPHY_VERBOSE );
#  endif


   const real Eint = (real)1.0e4*Pres;    // in principle, it can be set rather arbitrarily since Eint should be useless anyway
                                          // --> but still better to have reasonably large Eint to avoid error messages about
                                          //     Eint<0 during evolution

   return Eint;

} // FUNCTION : EoS_DensPres2Eint_Isothermal



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensPres2CSqr_Isothermal
// Description :  Convert gas mass density and pressure to sound speed squared
//
// Note        :  1. See EoS_DensEint2Pres_Isothermal()
//
// Parameter   :  Dens       : Gas mass density
//                Pres       : Gas pressure
//                Passive    : Passive scalars (must not used here)
//                AuxArray_* : Auxiliary arrays (see the Note above)
//                Table      : EoS tables
//
// Return      :  Sound speed squared
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensPres2CSqr_Isothermal( const real Dens, const real Pres, const real Passive[],
                                          const double AuxArray_Flt[], const int AuxArray_Int[],
                                          const real *const Table[EOS_NTABLE_MAX] )
{

// check
#  ifdef GAMER_DEBUG
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
#  endif


   const real Cs2 = AuxArray_Flt[0];

   return Cs2;

} // FUNCTION : EoS_DensPres2CSqr_Isothermal



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensEint2Temp_Isothermal
// Description :  Convert gas mass density and internal energy density to gas temperature
//
// Note        :  1. Internal energy density here is per unit volume instead of per unit mass
//                2. See EoS_SetAuxArray_Isothermal() for the values stored in AuxArray_Flt/Int[]
//                3. Temperature is in kelvin
//
// Parameter   :  Dens       : Gas mass density
//                Eint       : Gas internal energy density
//                Passive    : Passive scalars (must not used here)
//                AuxArray_* : Auxiliary arrays (see the Note above)
//                Table      : EoS tables
//
// Return      :  Gas temperature in kelvin
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensEint2Temp_Isothermal( const real Dens, const real Eint, const real Passive[],
                                          const double AuxArray_Flt[], const int AuxArray_Int[],
                                          const real *const Table[EOS_NTABLE_MAX] )
{

// check
#  ifdef GAMER_DEBUG
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
#  endif


   const real Temp = AuxArray_Flt[1];

   return Temp;

} // FUNCTION : EoS_DensEint2Temp_Isothermal



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensTemp2Pres_Isothermal
// Description :  Convert gas mass density and temperature to gas pressure
//
// Note        :  1. See EoS_SetAuxArray_Isothermal() for the values stored in AuxArray_Flt/Int[]
//                2. Temperature is in kelvin
//
// Parameter   :  Dens       : Gas mass density
//                Temp       : Gas temperature in kelvin
//                Passive    : Passive scalars (must not used here)
//                AuxArray_* : Auxiliary arrays (see the Note above)
//                Table      : EoS tables
//
// Return      :  Gas pressure
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensTemp2Pres_Isothermal( const real Dens, const real Temp, const real Passive[],
                                          const double AuxArray_Flt[], const int AuxArray_Int[],
                                          const real *const Table[EOS_NTABLE_MAX] )
{

// check
#  ifdef GAMER_DEBUG
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );

   Hydro_IsUnphysical_Single( Dens, "input density", TINY_NUMBER, HUGE_NUMBER, ERROR_INFO, UNPHY_VERBOSE );
#  endif


   const real Cs2  = AuxArray_Flt[0];
   const real Pres = Cs2*Dens;

   return Pres;

} // FUNCTION : EoS_DensTemp2Pres_Isothermal



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensEint2Entr_Isothermal
// Description :  Convert gas mass density and internal energy density to gas entropy
//
// Note        :  1. See EoS_SetAuxArray_Isothermal() for the values stored in AuxArray_Flt/Int[]
//
// Parameter   :  Dens       : Gas mass density
//                Eint       : Gas internal energy density
//                Passive    : Passive scalars (must not used here)
//                AuxArray_* : Auxiliary arrays (see the Note above)
//                Table      : EoS tables
//
// Return      :  Gas entropy
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensEint2Entr_Isothermal( const real Dens, const real Eint, const real Passive[],
                                          const double AuxArray_Flt[], const int AuxArray_Int[],
                                          const real *const Table[EOS_NTABLE_MAX] )
{

// EoS_DensEint2Entr is NOT supported yet for isothermal EoS
   return NULL_REAL;

} // FUNCTION : EoS_DensEint2Entr_Isothermal



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_General_Isothermal
// Description :  General EoS converter: In_*[] -> Out[]
//
// Note        :  1. See EoS_DensEint2Pres_Isothermal()
//                2. In_*[] and Out[] must NOT overlap
//                3. Useless for this EoS
//
// Parameter   :  Mode       : To support multiple modes in this general converter
//                Out        : Output array
//                In_*       : Input array
//                AuxArray_* : Auxiliary arrays (see the Note above)
//                Table      : EoS tables
//
// Return      :  Out[]
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static void EoS_General_Isothermal( const int Mode, real Out[], const real In_Flt[], const int In_Int[],
                                    const double AuxArray_Flt[], const int AuxArray_Int[],
                                    const real *const Table[EOS_NTABLE_MAX] )
{

// not used by this EoS

} // FUNCTION : EoS_General_Isothermal



// =============================================
// III. Set EoS initialization functions
// =============================================

#ifdef SYCL_LANGUAGE_VERSION
#  define FUNC_SPACE static
#else
#  define FUNC_SPACE            static
#endif

static dpct::global_memory<EoS_DE2P_t, 0>
    EoS_DensEint2Pres_Ptr(EoS_DensEint2Pres_Isothermal);
static dpct::global_memory<EoS_DP2E_t, 0>
    EoS_DensPres2Eint_Ptr(EoS_DensPres2Eint_Isothermal);
static dpct::global_memory<EoS_DP2C_t, 0>
    EoS_DensPres2CSqr_Ptr(EoS_DensPres2CSqr_Isothermal);
static dpct::global_memory<EoS_DE2T_t, 0>
    EoS_DensEint2Temp_Ptr(EoS_DensEint2Temp_Isothermal);
static dpct::global_memory<EoS_DT2P_t, 0>
    EoS_DensTemp2Pres_Ptr(EoS_DensTemp2Pres_Isothermal);
static dpct::global_memory<EoS_DE2S_t, 0>
    EoS_DensEint2Entr_Ptr(EoS_DensEint2Entr_Isothermal);
static dpct::global_memory<EoS_GENE_t, 0>
    EoS_General_Ptr(EoS_General_Isothermal);

//-----------------------------------------------------------------------------------------
// Function    :  EoS_SetCPU/GPUFunc_Isothermal
// Description :  Return the function pointers of the CPU/GPU EoS routines
//
// Note        :  1. Invoked by EoS_Init_Isothermal()
//                2. Must obtain the CPU and GPU function pointers by **separate** routines
//                   since CPU and GPU functions are compiled completely separately in GAMER
//                   --> In other words, a unified routine like the following won't work
//
//                      EoS_SetFunc_Isothermal( CPU_FuncPtr, GPU_FuncPtr );
//
//                3. Call-by-reference
//
// Parameter   :  EoS_DensEint2Pres_CPU/GPUPtr : CPU/GPU function pointers to be set
//                EoS_DensPres2Eint_CPU/GPUPtr : ...
//                EoS_DensPres2CSqr_CPU/GPUPtr : ...
//                EoS_DensEint2Temp_CPU/GPUPtr : ...
//                EoS_DensTemp2Pres_CPU/GPUPtr : ...
//                EoS_DensEint2Entr_CPU/GPUPtr : ...
//                EoS_General_CPU/GPUPtr       : ...
//
// Return      :  EoS_DensEint2Pres_CPU/GPUPtr, EoS_DensPres2Eint_CPU/GPUPtr,
//                EoS_DensPres2CSqr_CPU/GPUPtr, EoS_DensEint2Temp_CPU/GPUPtr,
//                EoS_DensTemp2Pres_CPU/GPUPtr, EoS_DensEint2Entr_CPU/GPUPtr,
//                EoS_General_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef SYCL_LANGUAGE_VERSION

void EoS_SetGPUFunc_Isothermal( EoS_DE2P_t &EoS_DensEint2Pres_GPUPtr,
                                EoS_DP2E_t &EoS_DensPres2Eint_GPUPtr,
                                EoS_DP2C_t &EoS_DensPres2CSqr_GPUPtr,
                                EoS_DE2T_t &EoS_DensEint2Temp_GPUPtr,
                                EoS_DT2P_t &EoS_DensTemp2Pres_GPUPtr,
                                EoS_DE2S_t &EoS_DensEint2Entr_GPUPtr,
                                EoS_GENE_t &EoS_General_GPUPtr )
{
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(&EoS_DensEint2Pres_GPUPtr, EoS_DensEint2Pres_Ptr.get_ptr(),
                   sizeof(EoS_DE2P_t))
           .wait()));
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(&EoS_DensPres2Eint_GPUPtr, EoS_DensPres2Eint_Ptr.get_ptr(),
                   sizeof(EoS_DP2E_t))
           .wait()));
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(&EoS_DensPres2CSqr_GPUPtr, EoS_DensPres2CSqr_Ptr.get_ptr(),
                   sizeof(EoS_DP2C_t))
           .wait()));
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(&EoS_DensEint2Temp_GPUPtr, EoS_DensEint2Temp_Ptr.get_ptr(),
                   sizeof(EoS_DE2T_t))
           .wait()));
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(&EoS_DensTemp2Pres_GPUPtr, EoS_DensTemp2Pres_Ptr.get_ptr(),
                   sizeof(EoS_DT2P_t))
           .wait()));
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(&EoS_DensEint2Entr_GPUPtr, EoS_DensEint2Entr_Ptr.get_ptr(),
                   sizeof(EoS_DE2S_t))
           .wait()));
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(&EoS_General_GPUPtr,
                                                 EoS_General_Ptr.get_ptr(),
                                                 sizeof(EoS_GENE_t))
                                         .wait()));
}

#else // #ifdef __CUDACC__

void EoS_SetCPUFunc_Isothermal( EoS_DE2P_t &EoS_DensEint2Pres_CPUPtr,
                                EoS_DP2E_t &EoS_DensPres2Eint_CPUPtr,
                                EoS_DP2C_t &EoS_DensPres2CSqr_CPUPtr,
                                EoS_DE2T_t &EoS_DensEint2Temp_CPUPtr,
                                EoS_DT2P_t &EoS_DensTemp2Pres_CPUPtr,
                                EoS_DE2S_t &EoS_DensEint2Entr_CPUPtr,
                                EoS_GENE_t &EoS_General_CPUPtr )
{
   EoS_DensEint2Pres_CPUPtr = EoS_DensEint2Pres_Ptr;
   EoS_DensPres2Eint_CPUPtr = EoS_DensPres2Eint_Ptr;
   EoS_DensPres2CSqr_CPUPtr = EoS_DensPres2CSqr_Ptr;
   EoS_DensEint2Temp_CPUPtr = EoS_DensEint2Temp_Ptr;
   EoS_DensTemp2Pres_CPUPtr = EoS_DensTemp2Pres_Ptr;
   EoS_DensEint2Entr_CPUPtr = EoS_DensEint2Entr_Ptr;
   EoS_General_CPUPtr       = EoS_General_Ptr;
}

#endif // #ifdef __CUDACC__ ... else ...

#ifndef SYCL_LANGUAGE_VERSION

// local function prototypes
void EoS_SetAuxArray_Isothermal( double [], int [] );
void EoS_SetCPUFunc_Isothermal( EoS_DE2P_t &, EoS_DP2E_t &, EoS_DP2C_t &, EoS_DE2T_t &, EoS_DT2P_t &, EoS_DE2S_t &, EoS_GENE_t & );
#ifdef GPU
void EoS_SetGPUFunc_Isothermal( EoS_DE2P_t &, EoS_DP2E_t &, EoS_DP2C_t &, EoS_DE2T_t &, EoS_DT2P_t &, EoS_DE2S_t &, EoS_GENE_t & );
#endif

//-----------------------------------------------------------------------------------------
// Function    :  EoS_Init_Isothermal
// Description :  Initialize EoS
//
// Note        :  1. Set auxiliary arrays by invoking EoS_SetAuxArray_*()
//                   --> It will be copied to GPU automatically in CUAPI_SetConstMemory()
//                2. Set the CPU/GPU EoS routines by invoking EoS_SetCPU/GPUFunc_*()
//                3. Invoked by EoS_Init()
//                   --> Enable it by linking to the function pointer "EoS_Init_Ptr"
//                4. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void EoS_Init_Isothermal()
{

// check
#  ifndef BAROTROPIC_EOS
   Aux_Error( ERROR_INFO, "must enable BAROTROPIC_EOS in the Makefile for the isothermal EoS !!\n" );
#  endif

   EoS_SetAuxArray_Isothermal( EoS_AuxArray_Flt, EoS_AuxArray_Int );
   EoS_SetCPUFunc_Isothermal( EoS_DensEint2Pres_CPUPtr, EoS_DensPres2Eint_CPUPtr,
                              EoS_DensPres2CSqr_CPUPtr, EoS_DensEint2Temp_CPUPtr,
                              EoS_DensTemp2Pres_CPUPtr, EoS_DensEint2Entr_CPUPtr,
                              EoS_General_CPUPtr );
#  ifdef GPU
   EoS_SetGPUFunc_Isothermal( EoS_DensEint2Pres_GPUPtr, EoS_DensPres2Eint_GPUPtr,
                              EoS_DensPres2CSqr_GPUPtr, EoS_DensEint2Temp_GPUPtr,
                              EoS_DensTemp2Pres_GPUPtr, EoS_DensEint2Entr_GPUPtr,
                              EoS_General_GPUPtr );
#  endif

} // FUNCTION : EoS_Init_Isothermal

#endif // #ifndef __CUDACC__



#endif // #if ( MODEL == HYDRO )
