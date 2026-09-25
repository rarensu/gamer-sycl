#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GPUAPI.h"
#include "CUDA_ConstMemory.h"

#if ( defined GPU  &&  MODEL == HYDRO )

extern real *d_EoS_Table[EOS_NTABLE_MAX];




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_SetConstMemory_EoS
// Description :  Set the EoS constant memory variables on GPU
//
// Note        :  1. Adopt the suggested approach for CUDA version >= 5.0
//                2. Invoked by EoS_Init()
//
// Parameter   :  None
//
// Return      :  c_EoS_AuxArray_Flt[], c_EoS_AuxArray_Int[], c_EoS_Table[]
//                EoS.AuxArrayDevPtr_Flt, EoS.AuxArrayDevPtr_Int, EoS.Table
//---------------------------------------------------------------------------------------------------
void CUAPI_SetConstMemory_EoS()
{

// copy data to constant memory
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(c_EoS_AuxArray_Flt.get_ptr(),
                                                 EoS_AuxArray_Flt,
                                                 EOS_NAUX_MAX * sizeof(double))
                                         .wait()));
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(c_EoS_AuxArray_Int.get_ptr(),
                                                 EoS_AuxArray_Int,
                                                 EOS_NAUX_MAX * sizeof(int))
                                         .wait()));
   DEVICE_CHECK_ERROR(
       DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(c_EoS_Table.get_ptr(), d_EoS_Table,
                                    EOS_NTABLE_MAX * sizeof(real *))
                            .wait()));

// obtain the constant-memory pointers
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(*((void **)&EoS.AuxArrayDevPtr_Flt) =
                                         c_EoS_AuxArray_Flt.get_ptr()));
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(*((void **)&EoS.AuxArrayDevPtr_Int) =
                                         c_EoS_AuxArray_Int.get_ptr()));
   DEVICE_CHECK_ERROR(
       DPCT_CHECK_ERROR(*((void **)&EoS.Table) = c_EoS_Table.get_ptr()));

} // FUNCTION : CUAPI_SetConstMemory_EoS



#endif // #if ( defined GPU  &&  MODEL == HYDRO )
