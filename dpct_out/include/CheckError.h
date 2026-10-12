#ifndef __DEVICE_CHECK_ERROR_H__
#define __DEVICE_CHECK_ERROR_H__
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Macro.h"

void Aux_Error( const char *File, const int Line, const char *Func, const char *Format, ... );


// SYCL error check — catches sycl::exception directly and reports the real
// error message (e.what()) through GAMER's Aux_Error, which prints a
// structured error banner and calls MPI_Exit() for proper MPI shutdown.
// This replaces the previous DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(expr))
// double-wrap, which caught the exception (discarding e.what()), converted
// it to a dpct::err0 code, then checked that code with a dummy string
// lookup via dpct::get_error_string_dummy().
#define DEVICE_CHECK_ERROR( Call ) \
  try { \
    Call; \
  } catch ( const sycl::exception &e ) { \
    Aux_Error( ERROR_INFO, "SYCL ERROR : %s !!\n", e.what() ); \
  }

// SYCL memory allocation error check
// Catches sycl::exception from sycl::malloc_device / sycl::malloc_host /
// dpct::dpct_malloc and reports the real error message via Aux_Error.
#define DEVICE_CHECK_MALLOC( Call ) \
{ \
  try { Call; } \
  catch ( const sycl::exception &e ) { \
    Aux_Error( ERROR_INFO, "SYCL ERROR in memory allocation : %s !!\n", e.what() ); \
    return GAMER_FAILED; \
  } \
}

#endif // #ifndef __DEVICE_CHECK_ERROR_H__
