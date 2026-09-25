#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GPUAPI.h"
#ifdef GPU

//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_Synchronize
// Description :  Block until the device has completed all preceding requested tasks
//
// Note        :  1. Replace the deprecated cudaThreadSynchronize() with cudaDeviceSynchronize()
//-------------------------------------------------------------------------------------------------------
void CUAPI_Synchronize()
{
   DEVICE_CHECK_ERROR(
       DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
}

#endif // #ifdef GPU
