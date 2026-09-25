#ifndef __CONSTMEMORY_H__
#define __CONSTMEMORY_H__

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Macro.h"
#include "Typedef.h"



#if ( MODEL == HYDRO )
inline dpct::constant_memory<double, 1> c_EoS_AuxArray_Flt(EOS_NAUX_MAX);
inline dpct::constant_memory<int, 1> c_EoS_AuxArray_Int(EOS_NAUX_MAX);
inline dpct::constant_memory<real *, 1> c_EoS_Table(EOS_NTABLE_MAX);
#endif

#if ( NCOMP_PASSIVE > 0 )
SET_GLOBAL( __constant__ int  c_NormIdx[NCOMP_PASSIVE] );
SET_GLOBAL( __constant__ int  c_FracIdx[NCOMP_PASSIVE] );
#else
inline dpct::constant_memory<int *, 0> c_NormIdx(NULL);
inline dpct::constant_memory<int *, 0> c_FracIdx(NULL);
#endif

#ifdef GRAVITY
inline dpct::constant_memory<double, 1> c_ExtAcc_AuxArray(EXT_ACC_NAUX_MAX);
inline dpct::constant_memory<double, 1> c_ExtPot_AuxArray_Flt(EXT_POT_NAUX_MAX);
inline dpct::constant_memory<int, 1> c_ExtPot_AuxArray_Int(EXT_POT_NAUX_MAX);

inline dpct::constant_memory<real, 1> c_Mp(3);
inline dpct::constant_memory<real, 1> c_Mm(3);
#endif

#if ( MODEL == HYDRO )
inline dpct::constant_memory<double, 1> c_Src_Dlep_AuxArray_Flt(SRC_NAUX_DLEP);
inline dpct::constant_memory<int, 1> c_Src_Dlep_AuxArray_Int(SRC_NAUX_DLEP);
#endif
#ifdef EXACT_COOLING
inline dpct::constant_memory<double, 1> c_Src_EC_AuxArray_Flt(SRC_NAUX_EC);
inline dpct::constant_memory<int, 1> c_Src_EC_AuxArray_Int(SRC_NAUX_EC);
#endif
inline dpct::constant_memory<double, 1> c_Src_User_AuxArray_Flt(SRC_NAUX_USER);
inline dpct::constant_memory<int, 1> c_Src_User_AuxArray_Int(SRC_NAUX_USER);

#endif // #ifndef __CONSTMEMORY_H__
