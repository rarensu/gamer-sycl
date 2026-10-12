#ifndef __CHECK_ERROR_H__
#define __CHECK_ERROR_H__



#include "Macro.h"

void Aux_Error( const char *File, const int Line, const char *Func, const char *Format, ... );

#ifdef SYCL_LANGUAGE_VERSION
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>


// SYCL error check — catches sycl::exception directly and reports the real
// error message (e.what()) through GAMER's Aux_Error, which prints a
// structured error banner and calls MPI_Exit() for proper MPI shutdown.
// This replaces the previous DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(expr))
// double-wrap, which caught the exception (discarding e.what()), converted
// it to a dpct::err0 code, then checked that code with a dummy string
// lookup via dpct::get_error_string_dummy().
// Wrapped in do-while(0) so the macro expands to a single statement.
// This is essential for correctness inside unbraced if/else chains: a
// bare "try {} catch {} ;" would introduce a spurious null statement that
// breaks the if-else linkage ("expected expression" at the following else).
#define DEVICE_CHECK_ERROR( Call ) \
  do { \
    try { \
      Call; \
    } catch ( const sycl::exception &e ) { \
      Aux_Error( ERROR_INFO, "SYCL ERROR : %s !!\n", e.what() ); \
    } \
  } while(0)

// SYCL memory allocation error check
// Catches sycl::exception from sycl::malloc_device / sycl::malloc_host /
// dpct::dpct_malloc and reports the real error message via Aux_Error.
#define DEVICE_CHECK_MALLOC( Call ) \
  do { \
    try { \
      Call; \
    } catch ( const sycl::exception &e ) { \
      Aux_Error( ERROR_INFO, "SYCL ERROR in memory allocation : %s !!\n", e.what() ); \
      return GAMER_FAILED; \
    } \
  } while(0)

#endif // SYCL_LANGUAGE_VERSION

#endif // #ifndef __CHECK_ERROR_H__
