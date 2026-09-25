#ifndef __CHECK_ERROR_H__
#define __CHECK_ERROR_H__



#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Macro.h"


void Aux_Error( const char *File, const int Line, const char *Func, const char *Format, ... );


// CUDA error check
#define DEVICE_CHECK_ERROR( Call )   DEVICE_Check_Error( Call, __FILE__, __LINE__, __FUNCTION__ )

inline void DEVICE_Check_Error( dpct::err0 Return, const char *File, const int Line, const char *Func )
{
   if ( Return != 0 )
      Aux_Error( File, Line, Func, "CUDA ERROR : %s !!\n", dpct::get_error_string_dummy(Return) );
}



// in DEVICE_CHECK_MALLOC(), we must use "Call; cudaError_t Return = cudaGetLastError();" instead of "cudaError_t Return = Call;"
// since cudaGetLastError() will reset the last error to cudaSuccess
// --> otherwise DEVICE_CHECK_ERROR( cudaGetLastError() ) in, for example, CUAPI_Asyn_FluidSolver(),
//     will fail since the last error has not been reset!
#define DEVICE_CHECK_MALLOC( Call )                                                                        \
{                                                                                                        \
   Call;                                                                                                 \
   const dpct::err0 Return = 0;                                                                          \
   if      ( Return == 2 )                                                                               \
      return GAMER_FAILED;                                                                               \
   else if ( Return != 0 )                                                                               \
      Aux_Error( ERROR_INFO, "CUDA ERROR in memory allocation : %s !!\n", dpct::get_error_string_dummy(Return) );  \
}



#endif // #ifndef __CHECK_ERROR_H__
