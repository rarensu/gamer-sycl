#include "GPUAPI.h"
#include "FLU.h"
#ifdef GRAVITY
#include "POT.h"
#endif

#ifdef GPU



// fluid solver prototypes in different models
#if   ( MODEL == HYDRO )
#if   ( FLU_SCHEME == RTVD )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
#else
__global__
#endif
void GPU_FluidSolver_RTVD(
   real g_Fluid_In [][NCOMP_TOTAL][ CUBE(FLU_NXT) ],
   real g_Fluid_Out[][NCOMP_TOTAL][ CUBE(PS2) ],
   real g_Flux     [][9][NCOMP_TOTAL][ SQR(PS2) ],
   const double g_Corner[][3],
   const real g_Pot_USG[][ CUBE(USG_NXT_F) ],
   const real dt, const real _dh, const bool StoreFlux,
   const bool XYZ, const real MinDens, const real MinPres, const real MinEint, const long PassiveFloor,
   const EoS_t EoS );
#elif ( FLU_SCHEME == MHM  ||  FLU_SCHEME == MHM_RP )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
#else
__global__
#endif
void GPU_FluidSolver_MHM(
   const real   g_Flu_Array_In [][NCOMP_TOTAL][ CUBE(FLU_NXT) ],
         real   g_Flu_Array_Out[][NCOMP_TOTAL][ CUBE(PS2) ],
   const real   g_Mag_Array_In [][NCOMP_MAG][ FLU_NXT_P1*SQR(FLU_NXT) ],
         real   g_Mag_Array_Out[][NCOMP_MAG][ PS2P1*SQR(PS2) ],
         char   g_DE_Array_Out [][ CUBE(PS2) ],
         real   g_Flux_Array   [][9][NCOMP_TOTAL][ SQR(PS2) ],
         real   g_Ele_Array    [][9][NCOMP_ELE][ PS2P1*PS2 ],
   const double g_Corner_Array [][3],
   const real   g_Pot_Array_USG[][ CUBE(USG_NXT_F) ],
         real   g_PriVar       []   [NCOMP_LR            ][ CUBE(FLU_NXT) ],
         real   g_Slope_PPM    [][3][NCOMP_LR            ][ CUBE(N_SLOPE_PPM) ],
         real   g_FC_Var       [][6][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_VAR) ],
         real   g_FC_Flux      [][3][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_FLUX) ],
         real   g_FC_Mag_Half  [][NCOMP_MAG][ FLU_NXT_P1*SQR(FLU_NXT) ],
         real   g_EC_Ele       [][NCOMP_MAG][ CUBE(N_EC_ELE) ],
   const real dt, const real dh,
   const bool StoreFlux, const bool StoreElectric,
   const LR_Limiter_t LR_Limiter, const real MinMod_Coeff, const int MinMod_MaxIter, const double Time,
   const bool UsePot, const OptExtAcc_t ExtAcc, const ExtAcc_t ExtAcc_Func,
   const real MinDens, const real MinPres, const real MinEint,
   const real DualEnergySwitch,
   const long PassiveFloor,
   const bool NormPassive, const int NNorm,
   const bool FracPassive, const int NFrac,
   #ifdef SYCL_LANGUAGE_VERSION
   const EoS_t EoS, const MicroPhy_t MicroPhy, int *const c_NormIdx, int *const c_FracIdx );
#else
   const EoS_t EoS, const MicroPhy_t MicroPhy );
#endif
#elif ( FLU_SCHEME == CTU )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
#else
__global__
#endif
void GPU_FluidSolver_CTU(
   const real   g_Flu_Array_In [][NCOMP_TOTAL][ CUBE(FLU_NXT) ],
         real   g_Flu_Array_Out[][NCOMP_TOTAL][ CUBE(PS2) ],
   const real   g_Mag_Array_In [][NCOMP_MAG][ FLU_NXT_P1*SQR(FLU_NXT) ],
         real   g_Mag_Array_Out[][NCOMP_MAG][ PS2P1*SQR(PS2) ],
         char   g_DE_Array_Out [][ CUBE(PS2) ],
         real   g_Flux_Array   [][9][NCOMP_TOTAL][ SQR(PS2) ],
         real   g_Ele_Array    [][9][NCOMP_ELE][ PS2P1*PS2 ],
   const double g_Corner_Array [][3],
   const real   g_Pot_Array_USG[][ CUBE(USG_NXT_F) ],
         real   g_PriVar       []   [NCOMP_LR            ][ CUBE(FLU_NXT) ],
         real   g_Slope_PPM    [][3][NCOMP_LR            ][ CUBE(N_SLOPE_PPM) ],
         real   g_FC_Var       [][6][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_VAR) ],
         real   g_FC_Flux      [][3][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_FLUX) ],
         real   g_FC_Mag_Half  [][NCOMP_MAG][ FLU_NXT_P1*SQR(FLU_NXT) ],
         real   g_EC_Ele       [][NCOMP_MAG][ CUBE(N_EC_ELE) ],
   const real dt, const real dh,
   const bool StoreFlux, const bool StoreElectric,
   const LR_Limiter_t LR_Limiter, const real MinMod_Coeff, const double Time,
   const bool UsePot, const OptExtAcc_t ExtAcc, const ExtAcc_t ExtAcc_Func,
   const real MinDens, const real MinPres, const real MinEint,
   const real DualEnergySwitch,
   const long PassiveFloor,
   const bool NormPassive, const int NNorm,
   const bool FracPassive, const int NFrac,
   #ifdef SYCL_LANGUAGE_VERSION
   const EoS_t EoS, int *const c_NormIdx, int *const c_FracIdx );
#else
   const EoS_t EoS );
#endif
#endif // FLU_SCHEME
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
void GPU_dtSolver_HydroCFL( real g_dt_Array[], const real g_Flu_Array[][FLU_NIN_T][ CUBE(PS1) ],
                                         const real g_Mag_Array[][NCOMP_MAG][ PS1P1*SQR(PS1) ],
                                         const real dh, const real Safety, const real MinPres,
                                         const long PassiveFloor, const EoS_t EoS, const MicroPhy_t MicroPhy,
                                         real *shared );
#else
__global__
void GPU_dtSolver_HydroCFL( real g_dt_Array[], const real g_Flu_Array[][FLU_NIN_T][ CUBE(PS1) ],
                                         const real g_Mag_Array[][NCOMP_MAG][ PS1P1*SQR(PS1) ],
                                         const real dh, const real Safety, const real MinPres,
                                         const long PassiveFloor, const EoS_t EoS, const MicroPhy_t MicroPhy );
#endif
#ifdef GRAVITY
#ifdef SYCL_LANGUAGE_VERSION
void GPU_dtSolver_HydroGravity( real g_dt_Array[], const real g_Pot_Array[][ CUBE(GRA_NXT) ],
                                  const double g_Corner_Array[][3],
                                  const real dh, const real Safety, const bool P5_Gradient,
                                  const bool UsePot, const OptExtAcc_t ExtAcc, const ExtAcc_t ExtAcc_Func,
                                  const double ExtAcc_Time );
#else
__global__
void GPU_dtSolver_HydroGravity( real g_dt_Array[], const real g_Pot_Array[][ CUBE(GRA_NXT) ],
                                  const double g_Corner_Array[][3],
                                  const real dh, const real Safety, const bool P5_Gradient,
                                  const bool UsePot, const OptExtAcc_t ExtAcc, const ExtAcc_t ExtAcc_Func,
                                  const double ExtAcc_Time );
#endif
#endif

#elif ( MODEL == ELBDM )
# if   ( WAVE_SCHEME == WAVE_FD )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
void GPU_ELBDMSolver_FD( real g_Fluid_In [][FLU_NIN ][ CUBE(FLU_NXT) ],
                                      real g_Fluid_Out[][FLU_NOUT][ CUBE(PS2) ],
                                      real g_Flux     [][9][NFLUX_TOTAL][ SQR(PS2) ],
                                      const real dt, const real _dh, const real Eta, const bool StoreFlux,
                                      const real Taylor3_Coeff, const bool XYZ, const real MinDens );
#else
__global__ void GPU_ELBDMSolver_FD( real g_Fluid_In [][FLU_NIN ][ CUBE(FLU_NXT) ],
                                      real g_Fluid_Out[][FLU_NOUT][ CUBE(PS2) ],
                                      real g_Flux     [][9][NFLUX_TOTAL][ SQR(PS2) ],
                                      const real dt, const real _dh, const real Eta, const bool StoreFlux,
                                      const real Taylor3_Coeff, const bool XYZ, const real MinDens );
#endif
# elif ( WAVE_SCHEME == WAVE_GRAMFE )
#  if   ( GRAMFE_SCHEME == GRAMFE_FFT )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
void GPU_ELBDMSolver_GramFE_FFT( real g_Fluid_In [][FLU_NIN ][ CUBE(FLU_NXT) ],
                                              real g_Fluid_Out[][FLU_NOUT ][ CUBE(PS2) ],
                                              real g_Flux     [][9][NFLUX_TOTAL][ SQR(PS2) ],
                                              const real dt, const real _dh, const real Eta, const bool StoreFlux,
                                              const bool XYZ, const real MinDens,
                                              typename FFT::workspace_type workspace,
                                              typename IFFT::workspace_type workspace_inverse );
#else
__launch_bounds__(FFT::max_threads_per_block)
__global__
void GPU_ELBDMSolver_GramFE_FFT( real g_Fluid_In [][FLU_NIN ][ CUBE(FLU_NXT) ],
                                              real g_Fluid_Out[][FLU_NOUT ][ CUBE(PS2) ],
                                              real g_Flux     [][9][NFLUX_TOTAL][ SQR(PS2) ],
                                              const real dt, const real _dh, const real Eta, const bool StoreFlux,
                                              const bool XYZ, const real MinDens,
                                              typename FFT::workspace_type workspace,
                                              typename IFFT::workspace_type workspace_inverse );
#endif
#  elif ( GRAMFE_SCHEME == GRAMFE_MATMUL )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
void GPU_ELBDMSolver_GramFE_MATMUL( real g_Fluid_In [][FLU_NIN ][ CUBE(FLU_NXT) ],
                                                 real g_Fluid_Out[][FLU_NOUT][ CUBE(PS2) ],
                                                 real g_Flux     [][9][NFLUX_TOTAL][ SQR(PS2) ],
                                                 gramfe_matmul_float g_Evolve[][ FLU_NXT*2 ],
                                                 const real dt, const real _dh, const real Eta, const bool StoreFlux,
                                                 const bool XYZ, const real MinDens );
#else
__global__
void GPU_ELBDMSolver_GramFE_MATMUL( real g_Fluid_In [][FLU_NIN ][ CUBE(FLU_NXT) ],
                                                 real g_Fluid_Out[][FLU_NOUT][ CUBE(PS2) ],
                                                 real g_Flux     [][9][NFLUX_TOTAL][ SQR(PS2) ],
                                                 gramfe_matmul_float g_Evolve[][ FLU_NXT*2 ],
                                                 const real dt, const real _dh, const real Eta, const bool StoreFlux,
                                                 const bool XYZ, const real MinDens );
#endif
#  else // GRAMFE_SCHEME
#     error : ERROR : unsupported GRAMFE_SCHEME !!
#  endif // GRAMFE_SCHEME
# else // WAVE_SCHEME
#  error : ERROR : unsupported WAVE_SCHEME !!
# endif // WAVE_SCHEME

#if ( ELBDM_SCHEME == ELBDM_HYBRID )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
void GPU_ELBDMSolver_HamiltonJacobi( real g_Fluid_In [][FLU_NIN ][ CUBE(HYB_NXT) ],
                                                  #ifdef GAMER_DEBUG
                                                  real g_Fluid_Out[][FLU_NOUT][ CUBE(PS2) ],
                                                  #else
                                                  real g_Fluid_Out[][FLU_NIN ][ CUBE(PS2) ],
                                                  #endif
                                                  real g_Flux     [][9][NFLUX_TOTAL][ SQR(PS2) ],
                                                  const bool g_IsCompletelyRefined[],
                                                                                                    const bool g_HasWaveCounterpart[][ CUBE(HYB_NXT) ],
                                                  const real dt, const real _dh, const real Eta, const bool StoreFlux,
                                                  const bool XYZ, const real MinDens );
#else
__global__ void GPU_ELBDMSolver_HamiltonJacobi( real g_Fluid_In [][FLU_NIN ][ CUBE(HYB_NXT) ],
                                                  #ifdef GAMER_DEBUG
                                                  real g_Fluid_Out[][FLU_NOUT][ CUBE(PS2) ],
                                                  #else
                                                  real g_Fluid_Out[][FLU_NIN ][ CUBE(PS2) ],
                                                  #endif
                                                  real g_Flux     [][9][NFLUX_TOTAL][ SQR(PS2) ],
                                                  const bool g_IsCompletelyRefined[],
                                                  const bool g_HasWaveCounterpart[][ CUBE(HYB_NXT) ],
                                                  const real dt, const real _dh, const real Eta, const bool StoreFlux,
                                                  const bool XYZ, const real MinDens );
#endif
#endif // #if ( ELBDM_SCHEME == ELBDM_HYBRID )

#else // MODEL
#error : ERROR : unsupported MODEL !!
#endif // MODEL


#ifdef GRAVITY

// Poisson solver prototypes
#if   ( POT_SCHEME == SOR )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL void GPU_PoissonSolver_SOR(
#else
__global__ void GPU_PoissonSolver_SOR(
#endif
const real g_Rho_Array    [][ CUBE(RHO_NXT) ],
                                         const real g_Pot_Array_In [][ CUBE(POT_NXT) ],
                                               real g_Pot_Array_Out[][ CUBE(GRA_NXT) ],
                                         const int Min_Iter, const int Max_Iter, const real Omega_6,
                                         const real Const, const IntScheme_t IntScheme );
#elif ( POT_SCHEME == MG )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL void GPU_PoissonSolver_MG(
#else
__global__ void GPU_PoissonSolver_MG(
#endif
const real g_Rho_Array    [][ CUBE(RHO_NXT) ],
                                        const real g_Pot_Array_In [][ CUBE(POT_NXT) ],
                                              real g_Pot_Array_Out[][ CUBE(GRA_NXT) ],
                                        const real dh_Min, const int Max_Iter, const int NPre_Smooth,
                                        const int NPost_Smooth, const real Tolerated_Error, const real Poi_Coeff,
                                        const IntScheme_t IntScheme );
#endif // POT_SCHEME


// Gravity solver prototypes in different models
#if   ( MODEL == HYDRO )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
#else
__global__
#endif
void GPU_HydroGravitySolver(
         real   g_Flu_Array_New[][GRA_NIN][ CUBE(PS1) ],
   const real   g_Pot_Array_New[][ CUBE(GRA_NXT) ],
   const double g_Corner_Array [][3],
   const real   g_Pot_Array_USG[][ CUBE(USG_NXT_G) ],
   const real   g_Flu_Array_USG[][GRA_NIN-1][ CUBE(PS1) ],
         char   g_DE_Array     [][ CUBE(PS1) ],
   const real   g_Emag_Array   [][ CUBE(PS1) ],
   const real dt, const real dh, const bool P5_Gradient,
      const bool UsePot, const OptExtAcc_t ExtAcc, const ExtAcc_t ExtAcc_Func,
#ifdef SYCL_LANGUAGE_VERSION
   const double TimeNew, const double TimeOld, const real MinEint,
   real *s_pot_new
#ifdef UNSPLIT_GRAVITY
   , real *s_pot_old
#endif
   );
#else
   const double TimeNew, const double TimeOld, const real MinEint );
#endif

#elif ( MODEL == ELBDM )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
void GPU_ELBDMGravitySolver(
#else
__global__
void GPU_ELBDMGravitySolver(
#endif
       real g_Flu_Array[][GRA_NIN][ CUBE(PS1) ],
                               const real g_Pot_Array[][ CUBE(GRA_NXT) ],
                               const real EtaDt, const real dh, const real Lambda );

#if ( ELBDM_SCHEME == ELBDM_HYBRID )
#ifdef SYCL_LANGUAGE_VERSION
SYCL_EXTERNAL
void GPU_ELBDMGravitySolver_HamiltonJacobi(
#else
__global__
void GPU_ELBDMGravitySolver_HamiltonJacobi(
#endif
       real g_Flu_Array[][GRA_NIN][ CUBE(PS1) ],
                                              const real g_Pot_Array[][ CUBE(GRA_NXT) ],
                                              const real EtaDt, const real dh, const real Lambda );
#endif
#else
#error : ERROR : unsupported MODEL !!
#endif // MODEL

#endif // GRAVITY


// source-term solver prototype
#ifdef SYCL_LANGUAGE_VERSION
void GPU_SrcSolver_IterateAllCells(
#else
__global__
void GPU_SrcSolver_IterateAllCells(
#endif
   const real g_Flu_Array_In [][FLU_NIN_S ][ CUBE(SRC_NXT)           ],
         real g_Flu_Array_Out[][FLU_NOUT_S][ CUBE(PS1)               ],
   const real g_Mag_Array_In [][NCOMP_MAG ][ SRC_NXT_P1*SQR(SRC_NXT) ],
   const double g_Corner_Array[][3],
   const SrcTerms_t SrcTerms, const int NPatchGroup, const real dt, const real dh,
   const double TimeNew, const double TimeOld,
   const real MinDens, const real MinPres, const real MinEint, const long PassiveFloor, const EoS_t EoS );




//-------------------------------------------------------------------------------------------------------
// Function    :  GPU_SetCache
// Description :  Set cache preference
//
// Parameter   :
//-------------------------------------------------------------------------------------------------------
void GPU_SetCache()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );

// SYCL currently does not support configuring shared memory cache preference on devices.
// TODO: Replacing these with calls to set configurable cache preference once SYCL supports it.

// 1. fluid solver
#  if   ( MODEL == HYDRO )
#  if   ( FLU_SCHEME == RTVD )
   // TODO: set preference to `PreferShared` for kernel `GPU_FluidSolver_RTVD`
#  elif ( FLU_SCHEME == MHM )
   // TODO: set preference to `PreferL1` for kernel `GPU_FluidSolver_MHM`
#  elif ( FLU_SCHEME == MHM_RP )
   // TODO: set preference to `PreferL1` for kernel `GPU_FluidSolver_MHM`
#  elif ( FLU_SCHEME == CTU )
   // TODO: set preference to `PreferL1` for kernel `GPU_FluidSolver_CTU`
#  endif
   // TODO: set preference to `PreferShared` for kernel `GPU_dtSolver_HydroCFL`
#  ifdef GRAVITY
   // TODO: set preference to `PreferShared` for kernel `GPU_dtSolver_HydroGravity`
#  endif

#  elif ( MODEL == ELBDM )
#  if   ( WAVE_SCHEME == WAVE_FD )
   // TODO: set preference to `PreferShared` for kernel `GPU_ELBDMSolver_FD`
#  elif ( WAVE_SCHEME == WAVE_GRAMFE )
#   if   ( GRAMFE_SCHEME == GRAMFE_FFT )
   // TODO: set preference to `PreferShared` for kernel `GPU_ELBDMSolver_GramFE_FFT`
#   elif ( GRAMFE_SCHEME == GRAMFE_MATMUL )
   // TODO: set preference to `PreferShared` for kernel `GPU_ELBDMSolver_GramFE_MATMUL`
#   else // GRAMFE_SCHEME
#   error : ERROR : unsupported GRAMFE_SCHEME !!
#   endif // GRAMFE_SCHEME
#  else // WAVE_SCHEME
#  error : ERROR : unsupported WAVE_SCHEME !!
#  endif // WAVE_SCHEME
#  if ( ELBDM_SCHEME == ELBDM_HYBRID )
   // TODO: set preference to `PreferShared` for kernel `GPU_ELBDMSolver_HamiltonJacobi`
#  endif

#  else
#  error : ERROR : unsupported MODEL !!
#  endif // MODEL


#  ifdef GRAVITY

// 2. Poisson solver
#  if   ( POT_SCHEME == SOR )
   // TODO: set preference to `PreferShared` for kernel `GPU_PoissonSolver_SOR`
#  elif ( POT_SCHEME == MG )
   // TODO: set preference to `PreferShared` for kernel `GPU_PoissonSolver_MG`
#  endif // POT_SCHEME


// 3. gravity solver
#  if   ( MODEL == HYDRO )
   // TODO: set preference to `PreferShared` for kernel `GPU_HydroGravitySolver`

#  elif ( MODEL == ELBDM )
   // TODO: set preference to `PreferL1` for kernel `GPU_ELBDMGravitySolver`

#  else
#  error : ERROR : unsupported MODEL !!
#  endif // MODEL

#  endif // GRAVITY


// 4. source-term solver
   // TODO: set preference to `PreferL1` for kernel `GPU_SrcSolver_IterateAllCells`


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : GPU_SetCache



#endif // #ifdef GPU
