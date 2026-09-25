#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "FLU.h"

// external functions and GPU-related set-up
#ifdef SYCL_LANGUAGE_VERSION

#include "CUDA_CheckError.h"
#include "CUDA_ConstMemory.h"
#if ( MODEL == HYDRO )
#include "CPU_Shared_FluUtility.cpp"
#endif

#endif // #ifdef __CUDACC__


// local function prototypes
#ifndef SYCL_LANGUAGE_VERSION

void Src_SetAuxArray_User_Template( double [], int [] );
void Src_SetConstMemory_User_Template( const double AuxArray_Flt[], const int AuxArray_Int[],
                                       double *&DevPtr_Flt, int *&DevPtr_Int );
void Src_SetCPUFunc_User_Template( SrcFunc_t & );
#ifdef GPU
void Src_SetGPUFunc_User_Template( SrcFunc_t & );
#endif
void Src_WorkBeforeMajorFunc_User_Template( const int lv, const double TimeNew, const double TimeOld, const double dt,
                                            double AuxArray_Flt[], int AuxArray_Int[] );
void Src_End_User_Template();

#endif



/********************************************************
1. Template of a user-defined source term
   --> Enabled by the runtime option "SRC_USER"

2. This file is shared by both CPU and GPU

   CUSRC_Src_User_Template.cu -> CPU_Src_User_Template.cpp

3. Four steps are required to implement a source term

   I.   Set auxiliary arrays
   II.  Implement the source-term function
   III. [Optional] Add the work to be done every time
        before calling the major source-term function
   IV.  Set initialization functions

4. The source-term function must be thread-safe and
   not use any global variable
********************************************************/



// =======================
// I. Set auxiliary arrays
// =======================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetAuxArray_User_Template
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
// Note        :  1. Invoked by Src_Init_User_Template()
//                2. AuxArray_Flt/Int[] have the size of SRC_NAUX_USER defined in Macro.h (default = 10)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef SYCL_LANGUAGE_VERSION
void Src_SetAuxArray_User_Template( double AuxArray_Flt[], int AuxArray_Int[] )
{

   /*
   AuxArray_Flt[0] = ...;
   AuxArray_Flt[1] = ...;

   AuxArray_Int[0] = ...;
   AuxArray_Int[1] = ...;
   */

} // FUNCTION : Src_SetAuxArray_User_Template
#endif // #ifndef __CUDACC__



// ======================================
// II. Implement the source-term function
// ======================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_User_Template
// Description :  Major source-term function
//
// Note        :  1. Invoked by CPU/GPU_SrcSolver_IterateAllCells()
//                2. See Src_SetAuxArray_User_Template() for the values stored in AuxArray_Flt/Int[]
//                3. Shared by both CPU and GPU
//
// Parameter   :  fluid             : Fluid array storing both the input and updated values
//                                    --> Including both active and passive variables
//                B                 : Cell-centered magnetic field
//                SrcTerms          : Structure storing all source-term variables
//                dt                : Time interval to advance solution
//                dh                : Grid size
//                x/y/z             : Target physical coordinates
//                TimeNew           : Target physical time to reach
//                TimeOld           : Physical time before update
//                                    --> This function updates physical time from TimeOld to TimeNew
//                MinDens/Pres/Eint : Density, pressure, and internal energy floors
//                PassiveFloor      : Bitwise flag to specify the passive scalars to be floored
//                EoS               : EoS object
//                AuxArray_*        : Auxiliary arrays (see the Note above)
//
// Return      :  fluid[]
//-----------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static void Src_User_Template( real fluid[], const real B[],
                               const SrcTerms_t *SrcTerms, const real dt, const real dh,
                               const double x, const double y, const double z,
                               const double TimeNew, const double TimeOld,
                               const real MinDens, const real MinPres, const real MinEint, const long PassiveFloor,
                               const EoS_t *EoS, const double AuxArray_Flt[], const int AuxArray_Int[] )
{

// check
#  ifdef GAMER_DEBUG
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );
#  endif

// example
   /*
   const bool CheckMinEint_Yes = true;
   const real CoolingRate      = (real)AuxArray_Flt[0];

   real Eint, Enth, Emag;

#  ifdef MHD
   Emag  = (real)0.5*( SQR(B[MAGX]) + SQR(B[MAGY]) + SQR(B[MAGZ]) );
#  else
   Emag  = (real)0.0;
#  endif
   Eint  = Hydro_Con2Eint( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY],
                           CheckMinEint_Yes, MinEint, PassiveFloor, Emag, EoS->GuessHTilde_CPUPtr,
                           EoS->HTilde2Temp_CPUPtr, EoS->AuxArray_Flt, EoS->AuxArray_Int,
                           EoS->Table );
   Enth  = fluid[ENGY] - Eint;
   Eint -= fluid[DENS]*CoolingRate*dt;

   fluid[ENGY] = Enth + Eint;
   */

} // FUNCTION : Src_User_Template



// ==================================================
// III. [Optional] Add the work to be done every time
//      before calling the major source-term function
// ==================================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_WorkBeforeMajorFunc_User_Template
// Description :  Specify work to be done every time before calling the major source-term function
//
// Note        :  1. Invoked by Src_WorkBeforeMajorFunc()
//                   --> By linking to "Src_WorkBeforeMajorFunc_User_Ptr" in Src_Init_User_Template()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  lv               : Target refinement level
//                TimeNew          : Target physical time to reach
//                TimeOld          : Physical time before update
//                                   --> The major source-term function will update the system from TimeOld to TimeNew
//                dt               : Time interval to advance solution
//                                   --> Physical coordinates : TimeNew - TimeOld == dt
//                                       Comoving coordinates : TimeNew - TimeOld == delta(scale factor) != dt
//                AuxArray_Flt/Int : Auxiliary arrays
//                                   --> Can be used and/or modified here
//                                   --> Must call Src_SetConstMemory_User_Template() after modification
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef SYCL_LANGUAGE_VERSION
void Src_WorkBeforeMajorFunc_User_Template( const int lv, const double TimeNew, const double TimeOld, const double dt,
                                            double AuxArray_Flt[], int AuxArray_Int[] )
{

// uncomment the following lines if the auxiliary arrays have been modified
//#  ifdef GPU
//   Src_SetConstMemory_User_Template( AuxArray_Flt, AuxArray_Int,
//                                     SrcTerms.User_AuxArrayDevPtr_Flt, SrcTerms.User_AuxArrayDevPtr_Int );
//#  endif

} // FUNCTION : Src_WorkBeforeMajorFunc_User_Template
#endif



// ================================
// IV. Set initialization functions
// ================================

#ifdef SYCL_LANGUAGE_VERSION
#  define FUNC_SPACE static
#else
#  define FUNC_SPACE            static
#endif

static dpct::global_memory<SrcFunc_t, 0> SrcFunc_Ptr(Src_User_Template);

//-----------------------------------------------------------------------------------------
// Function    :  Src_SetCPU/GPUFunc_User_Template
// Description :  Return the function pointer of the CPU/GPU source-term function
//
// Note        :  1. Invoked by Src_Init_User_Template()
//                2. Call-by-reference
//
// Parameter   :  SrcFunc_CPU/GPUPtr : CPU/GPU function pointer to be set
//
// Return      :  SrcFunc_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef SYCL_LANGUAGE_VERSION

void Src_SetGPUFunc_User_Template( SrcFunc_t &SrcFunc_GPUPtr )
{
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(&SrcFunc_GPUPtr, SrcFunc_Ptr.get_ptr(), sizeof(SrcFunc_t))
           .wait()));
}

#else

void Src_SetCPUFunc_User_Template( SrcFunc_t &SrcFunc_CPUPtr )
{
   SrcFunc_CPUPtr = SrcFunc_Ptr;
}

#endif // #ifdef __CUDACC__ ... else ...

#ifdef SYCL_LANGUAGE_VERSION
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetConstMemory_User_Template
// Description :  Set the constant memory variables on GPU
//
// Note        :  1. Adopt the suggested approach for CUDA version >= 5.0
//                2. Invoked by Src_Init_User_Template() and, if necessary, Src_WorkBeforeMajorFunc_User_Template()
//                3. SRC_NAUX_USER is defined in Macro.h
//
// Parameter   :  AuxArray_Flt/Int : Auxiliary arrays to be copied to the constant memory
//                DevPtr_Flt/Int   : Pointers to store the addresses of constant memory arrays
//
// Return      :  c_Src_User_AuxArray_Flt[], c_Src_User_AuxArray_Int[], DevPtr_Flt, DevPtr_Int
//---------------------------------------------------------------------------------------------------
void Src_SetConstMemory_User_Template( const double AuxArray_Flt[], const int AuxArray_Int[],
                                       double *&DevPtr_Flt, int *&DevPtr_Int )
{

// copy data to constant memory
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(c_Src_User_AuxArray_Flt.get_ptr(), AuxArray_Flt,
                   SRC_NAUX_USER * sizeof(double))
           .wait()));
   CUDA_CHECK_ERROR(
       DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(c_Src_User_AuxArray_Int.get_ptr(),
                                    AuxArray_Int, SRC_NAUX_USER * sizeof(int))
                            .wait()));

// obtain the constant-memory pointers
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(*((void **)&DevPtr_Flt) =
                                         c_Src_User_AuxArray_Flt.get_ptr()));
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(*((void **)&DevPtr_Int) =
                                         c_Src_User_AuxArray_Int.get_ptr()));

} // FUNCTION : Src_SetConstMemory_User_Template
#endif // #ifdef __CUDACC__

#ifndef SYCL_LANGUAGE_VERSION

//-----------------------------------------------------------------------------------------
// Function    :  Src_Init_User_Template
// Description :  Initialize a user-specified source term
//
// Note        :  1. Set auxiliary arrays by invoking Src_SetAuxArray_*()
//                   --> Copy to the GPU constant memory and store the associated addresses
//                2. Set the source-term function by invoking Src_SetCPU/GPUFunc_*()
//                3. Set the function pointers "Src_WorkBeforeMajorFunc_User_Ptr" and "Src_End_User_Ptr"
//                4. Invoked by Src_Init()
//                   --> Enable it by linking to the function pointer "Src_Init_User_Ptr"
//                5. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_Init_User_Template()
{

// set the auxiliary arrays
   Src_SetAuxArray_User_Template( Src_User_AuxArray_Flt, Src_User_AuxArray_Int );

// copy the auxiliary arrays to the GPU constant memory and store the associated addresses
#  ifdef GPU
   Src_SetConstMemory_User_Template( Src_User_AuxArray_Flt, Src_User_AuxArray_Int,
                                     SrcTerms.User_AuxArrayDevPtr_Flt, SrcTerms.User_AuxArrayDevPtr_Int );
#  else
   SrcTerms.User_AuxArrayDevPtr_Flt = Src_User_AuxArray_Flt;
   SrcTerms.User_AuxArrayDevPtr_Int = Src_User_AuxArray_Int;
#  endif

// set the major source-term function
   Src_SetCPUFunc_User_Template( SrcTerms.User_CPUPtr );

#  ifdef GPU
   Src_SetGPUFunc_User_Template( SrcTerms.User_GPUPtr );
   SrcTerms.User_FuncPtr = SrcTerms.User_GPUPtr;
#  else
   SrcTerms.User_FuncPtr = SrcTerms.User_CPUPtr;
#  endif

// set the auxiliary functions
   Src_WorkBeforeMajorFunc_User_Ptr = Src_WorkBeforeMajorFunc_User_Template;
   Src_End_User_Ptr                 = Src_End_User_Template;

} // FUNCTION : Src_Init_User_Template



//-----------------------------------------------------------------------------------------
// Function    :  Src_End_User_Template
// Description :  Free the resources used by a user-specified source term
//
// Note        :  1. Invoked by Src_End()
//                   --> Enable it by linking to the function pointer "Src_End_User_Ptr"
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_End_User_Template()
{


} // FUNCTION : Src_End_User_Template

#endif // #ifndef __CUDACC__
