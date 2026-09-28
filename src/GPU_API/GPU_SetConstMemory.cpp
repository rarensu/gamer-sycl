// define DEFINE_GLOBAL to declare all constant variables here
// --> must define it BEFORE including GPUAPI.h since the latter will include "Macro.h" to set SET_GLOBAL()
#define DEFINE_GLOBAL
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GPUAPI.h"
#include "ConstMemory.h"
#undef DEFINE_GLOBAL

#ifdef GPU

extern real *d_EoS_Table[EOS_NTABLE_MAX];

#ifdef GRAVITY
void GPU_SetConstMemory_ExtAccPot();
#endif




//-------------------------------------------------------------------------------------------------------
// Function    :  GPU_SetConstMemory
// Description :  Set the constant memory variables on GPU
//
// Note        :  1. Adopt the suggested approach for SYCL
//                2. Invoked by Init_GAMER()
//                3. Invoke GPU_SetConstMemory_ExtAccPot()
//                4. Some constant memory variables are set elsewhere. For example,
//                   (1) Source-term variables are set by individual source-term initializer
//                   (2) EoS variables are set by GPU_SetConstMemory_EoS()
//
// Parameter   :  None
//
// Return      :  c_NormIdx[], c_FracIdx[], c_Mp[], c_Mm[], c_ExtAcc_AuxArray[], c_ExtPot_AuxArray[]
//---------------------------------------------------------------------------------------------------
void GPU_SetConstMemory()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// copy data to constant memory
#  if ( NCOMP_PASSIVE > 0 )
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(c_NormIdx.get_ptr(), PassiveNorm_VarIdx,
                   NCOMP_PASSIVE * sizeof(int))
           .wait()));
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(c_FracIdx.get_ptr(), PassiveIntFrac_VarIdx,
                   NCOMP_PASSIVE * sizeof(int))
           .wait()));
#  endif

#  ifdef GRAVITY
   GPU_SetConstMemory_ExtAccPot();

   const real h_Mp[3] = { -3.0/32.0, +30.0/32.0, +5.0/32.0 };
   const real h_Mm[3] = { +5.0/32.0, +30.0/32.0, -3.0/32.0 };

   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(c_Mp.get_ptr(), h_Mp, 3 * sizeof(real))
           .wait()));
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(c_Mm.get_ptr(), h_Mm, 3 * sizeof(real))
           .wait()));
#  endif // #ifdef GRAVITY


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : GPU_SetConstMemory



#endif // #ifdef GPU
