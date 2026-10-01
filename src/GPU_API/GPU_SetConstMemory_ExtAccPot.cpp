#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GPUAPI.h"
#include "ConstMemory.h"

#if ( defined GPU  &&  defined GRAVITY )




//-------------------------------------------------------------------------------------------------------
// Function    :  GPU_SetConstMemory_ExtAccPot
// Description :  Set the constant memory variables on GPU used by the external acceleration and
//                potential routines
//
// Note        :  1. Adopt the suggested approach for SYCL
//                2. Invoked by GPU_SetConstMemory()
//                3. EXT_ACC_NAUX_MAX and EXT_POT_NAUX_MAX are defined in Macro.h
//
// Parameter   :  None
//
// Return      :  c_ExtAcc_AuxArray[], c_ExtPot_AuxArray_Flt[], c_ExtPot_AuxArray_Int[]
//---------------------------------------------------------------------------------------------------
void GPU_SetConstMemory_ExtAccPot()
{

   if ( OPT__EXT_ACC )
      DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
          dpct::get_in_order_queue()
              .memcpy(c_ExtAcc_AuxArray.get_ptr(), ExtAcc_AuxArray,
                      EXT_ACC_NAUX_MAX * sizeof(double))
              .wait()));

   if ( OPT__EXT_POT ) {
      DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
          dpct::get_in_order_queue()
              .memcpy(c_ExtPot_AuxArray_Flt.get_ptr(), ExtPot_AuxArray_Flt,
                      EXT_POT_NAUX_MAX * sizeof(double))
              .wait()));
      DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
          dpct::get_in_order_queue()
              .memcpy(c_ExtPot_AuxArray_Int.get_ptr(), ExtPot_AuxArray_Int,
                      EXT_POT_NAUX_MAX * sizeof(int))
              .wait()));
   }

} // FUNCTION : GPU_SetConstMemory_ExtAccPot



#endif // #if ( defined GPU  &&  defined GRAVITY )
