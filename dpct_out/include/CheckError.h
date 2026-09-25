#ifndef __DEVICE_CHECK_ERROR_H__
#define __DEVICE_CHECK_ERROR_H__

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Macro.h"

void Aux_Error( const char *File, const int Line, const char *Func, const char *Format, ... );


// CUDA error check
#define DEVICE_CHECK_ERROR( Call )   DEVICE_Check_Error( Call, __FILE__, __LINE__, __FUNCTION__ )

inline void DEVICE_Check_Error(dpct::err0 Return, const char *File,
                             const int Line, const char *Func)
{
   /*
   DPCT1000:13: Error handling if-stmt was detected but could not be rewritten.
   */
   if (Return != 0)
      /*
      DPCT1009:14: SYCL reports errors using exceptions and does not use error
      codes. Please replace the "get_error_string_dummy(...)" with a real
      error-handling function.
      */
      /*
      DPCT1001:12: The statement could not be removed.
      */
      Aux_Error(File, Line, Func, "CUDA ERROR : %s !!\n",
                dpct::get_error_string_dummy(Return));
}



// in DEVICE_CHECK_MALLOC(), we must use "Call; cudaError_t Return = cudaGetLastError();" instead of "cudaError_t Return = Call;"
// since cudaGetLastError() will reset the last error to cudaSuccess
// --> otherwise DEVICE_CHECK_ERROR( cudaGetLastError() ) in, for example, CUAPI_Asyn_FluidSolver(),
//     will fail since the last error has not been reset!
/*
DPCT1010:66: SYCL uses exceptions to report errors and does not use the error
codes. The cudaGetLastError function call was replaced with 0. You need to
rewrite this code.
*/
/*
DPCT1001:67: The statement could not be removed.
*/
/*
DPCT1000:68: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1009:69: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define DEVICE_CHECK_MALLOC(Call)                                                \
   {                                                                           \
      Call;                                                                    \
      const dpct::err0 Return = 0;                                             \
      if (Return == 2)                                                         \
         return GAMER_FAILED;                                                  \
      else if (Return != 0)                                                    \
         Aux_Error(ERROR_INFO, "CUDA ERROR in memory allocation : %s !!\n",    \
                   dpct::get_error_string_dummy(Return));                      \
   }

#endif // #ifndef __DEVICE_CHECK_ERROR_H__
