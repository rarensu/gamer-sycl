#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GPUAPI.h"
#include "FLU.h"

#ifdef GPU



// ***************************************************
// ** GPU queue/stream objects are declared here **
dpct::queue_ptr *Stream;
// ***************************************************


extern real (*d_Flu_Array_F_In )[FLU_NIN ][ CUBE(FLU_NXT) ];
extern real (*d_Flu_Array_F_Out)[FLU_NOUT][ CUBE(PS2) ];
extern real (*d_Flux_Array)[9][NFLUX_TOTAL][ SQR(PS2) ];
#ifdef UNSPLIT_GRAVITY
extern real (*d_Pot_Array_USG_F)[ CUBE(USG_NXT_F) ];
extern double (*d_Corner_Array_F)[3];
#endif
#ifdef DUAL_ENERGY
extern char (*d_DE_Array_F_Out)[ CUBE(PS2) ];
#endif
#ifdef MHD
extern real (*d_Mag_Array_F_In )[NCOMP_MAG][ FLU_NXT_P1*SQR(FLU_NXT) ];
extern real (*d_Mag_Array_F_Out)[NCOMP_MAG][ PS2P1*SQR(PS2)          ];
extern real (*d_Ele_Array      )[9][NCOMP_ELE][ PS2P1*PS2 ];
extern real (*d_Mag_Array_T)[NCOMP_MAG][ PS1P1*SQR(PS1) ];
extern real (*d_Mag_Array_S_In)[NCOMP_MAG][ SRC_NXT_P1*SQR(SRC_NXT) ];
#endif
extern real *d_dt_Array_T;
extern real (*d_Flu_Array_T)[FLU_NIN_T][ CUBE(PS1) ];
extern real (*d_Flu_Array_S_In )[FLU_NIN_S ][ CUBE(SRC_NXT) ];
extern real (*d_Flu_Array_S_Out)[FLU_NOUT_S][ CUBE(PS1)     ];
extern double (*d_Corner_Array_S)[3];
#if ( FLU_SCHEME == MHM  ||  FLU_SCHEME == MHM_RP  ||  FLU_SCHEME == CTU )
extern real (*d_PriVar)      [NCOMP_LR            ][ CUBE(FLU_NXT)     ];
extern real (*d_Slope_PPM)[3][NCOMP_LR            ][ CUBE(N_SLOPE_PPM) ];
extern real (*d_FC_Var)   [6][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_VAR)    ];
extern real (*d_FC_Flux)  [3][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_FLUX)   ];
#ifdef MHD
extern real (*d_FC_Mag_Half)[NCOMP_MAG][ FLU_NXT_P1*SQR(FLU_NXT) ];
extern real (*d_EC_Ele     )[NCOMP_MAG][ CUBE(N_EC_ELE)          ];
#endif
#endif // FLU_SCHEME

#if ( MODEL == ELBDM )
extern bool (*d_IsCompletelyRefined);
#endif

#if ( ELBDM_SCHEME == ELBDM_HYBRID )
extern bool (*d_HasWaveCounterpart)[ CUBE(HYB_NXT) ];
#endif

#if ( GRAMFE_SCHEME == GRAMFE_MATMUL )
extern gramfe_matmul_float (*d_Flu_TimeEvo)[ 2*FLU_NXT ];
#endif

#if ( MODEL != HYDRO  &&  MODEL != ELBDM )
#  warning : DO YOU WANT TO ADD SOMETHING HERE FOR THE NEW MODEL ??
#endif




//-------------------------------------------------------------------------------------------------------
// Function    :  GPU_MemAllocate_Fluid
// Description :  Allocate GPU and CPU memory for the fluid solver
//
// Parameter   :  Flu_NPG     : Number of patch groups evaluated simultaneously by GPU for the fluid solver
//                Pot_NPG     : Number of patch groups evaluated simultaneously by GPU for the gravity solver
//                              --> Here it is used only for the dt solver
//                Src_NPG     : Number of patch groups evaluated simultaneously by GPU for the source-term solver
//                GPU_NStream : Number of GPU queue/stream objects
//
// Return      :  GAMER_SUCCESS / GAMER_FAILED
//-------------------------------------------------------------------------------------------------------
int GPU_MemAllocate_Fluid( const int Flu_NPG, const int Pot_NPG, const int Src_NPG, const int GPU_NStream )
{

// size of the global memory arrays in all models
   const int  Flu_NP                = 8*Flu_NPG;
#  ifdef GRAVITY
   const int  Pot_NP                = 8*Pot_NPG;
#  endif
   const int  Src_NP                = 8*Src_NPG;
   const long Flu_MemSize_F_In      = sizeof(real  )*Flu_NPG*FLU_NIN *CUBE(FLU_NXT);
   const long Flu_MemSize_F_Out     = sizeof(real  )*Flu_NPG*FLU_NOUT*CUBE(PS2);
   const long Flux_MemSize          = sizeof(real  )*Flu_NPG*9*NFLUX_TOTAL*SQR(PS2);
#  ifdef UNSPLIT_GRAVITY
   const long Pot_MemSize_USG_F     = sizeof(real  )*Flu_NPG*CUBE(USG_NXT_F);
   const long Corner_MemSize_F      = sizeof(double)*Flu_NPG*3;
#  endif
#  ifdef DUAL_ENERGY
   const long DE_MemSize_F_Out      = sizeof(char  )*Flu_NPG*CUBE(PS2);
#  endif
#  ifdef MHD
   const long Mag_MemSize_F_In      = sizeof(real  )*Flu_NPG*NCOMP_MAG*FLU_NXT_P1*SQR(FLU_NXT);
   const long Mag_MemSize_F_Out     = sizeof(real  )*Flu_NPG*NCOMP_MAG*PS2P1*SQR(PS2);
   const long Ele_MemSize           = sizeof(real  )*Flu_NPG*9*NCOMP_ELE*PS2P1*PS2;
   const long Mag_MemSize_T         = sizeof(real  )*Flu_NP*NCOMP_MAG*PS1P1*SQR(PS1);
   const long Mag_MemSize_S_In      = sizeof(real  )*Src_NP*NCOMP_MAG*SRC_NXT_P1*SQR(SRC_NXT);
#  endif
#  ifdef GRAVITY
   const long dt_MemSize_T          = sizeof(real  )*MAX( Flu_NP, Pot_NP ); // dt_Array_T is used for both DT_FLU_SOLVER and DT_GRA_SOLVER
#  else
   const long dt_MemSize_T          = sizeof(real  )*Flu_NP;
#  endif
   const long Flu_MemSize_T         = sizeof(real  )*Flu_NP*FLU_NIN_T *CUBE(PS1);
   const long Flu_MemSize_S_In      = sizeof(real  )*Src_NP*FLU_NIN_S *CUBE(SRC_NXT);
   const long Flu_MemSize_S_Out     = sizeof(real  )*Src_NP*FLU_NOUT_S*CUBE(PS1);
   const long Corner_MemSize_S      = sizeof(double)*Src_NP*3;
#  ifdef EXACT_COOLING
   const long EC_TEF_lambda_MemSize = sizeof(double)*SrcTerms.EC_TEF_N;
   const long EC_TEF_alpha_MemSize  = sizeof(double)*SrcTerms.EC_TEF_N;
   const long EC_TEFc_MemSize       = sizeof(double)*SrcTerms.EC_TEF_N;
#  endif

// the size of the global memory arrays in different models
#  if ( FLU_SCHEME == MHM  ||  FLU_SCHEME == MHM_RP  ||  FLU_SCHEME == CTU )
   const long PriVar_MemSize        = sizeof(real  )*Flu_NPG  *NCOMP_LR            *CUBE(FLU_NXT);
   const long FC_Var_MemSize        = sizeof(real  )*Flu_NPG*6*NCOMP_TOTAL_PLUS_MAG*CUBE(N_FC_VAR);
   const long FC_Flux_MemSize       = sizeof(real  )*Flu_NPG*3*NCOMP_TOTAL_PLUS_MAG*CUBE(N_FC_FLUX);
#  if ( LR_SCHEME == PPM )
   const long Slope_PPM_MemSize     = sizeof(real  )*Flu_NPG*3*NCOMP_LR            *CUBE(N_SLOPE_PPM);
#  endif
#  ifdef MHD
   const long FC_Mag_Half_MemSize   = sizeof(real  )*Flu_NPG  *NCOMP_MAG*FLU_NXT_P1*SQR(FLU_NXT);
   const long EC_Ele_MemSize        = sizeof(real  )*Flu_NPG  *NCOMP_MAG*CUBE(N_EC_ELE);
#  endif
#  endif // FLU_SCHEME

#  if ( MODEL == ELBDM )
   const long Flu_MemSize_IsCompletelyRefined = sizeof(bool )*Flu_NPG;
#  endif

#  if ( ELBDM_SCHEME == ELBDM_HYBRID )
   const long Flu_MemSize_HasWaveCounterpart  = sizeof(bool )*Flu_NPG*CUBE(HYB_NXT);
#  endif

#  if ( GRAMFE_SCHEME == GRAMFE_MATMUL )
   const long GramFE_TimeEvo_MemSize          = sizeof(gramfe_matmul_float)*2*PS2*FLU_NXT;
#  endif

#  if ( MODEL != HYDRO  &&  MODEL != ELBDM )
#     warning : DO YOU WANT TO ADD SOMETHING HERE FOR THE NEW MODEL ??
#  endif


// output the total memory requirement
   long TotalSize = Flu_MemSize_F_In + Flu_MemSize_F_Out + dt_MemSize_T + Flu_MemSize_T;

   if ( amr->WithFlux )
   TotalSize += Flux_MemSize;

#  ifdef UNSPLIT_GRAVITY
   TotalSize += Pot_MemSize_USG_F;

   if ( OPT__EXT_ACC )
   TotalSize += Corner_MemSize_F;
#  endif

#  ifdef DUAL_ENERGY
   TotalSize += DE_MemSize_F_Out;
#  endif

#  ifdef MHD
   TotalSize += Mag_MemSize_F_In + Mag_MemSize_F_Out + Mag_MemSize_T;

   if ( amr->WithElectric )
   TotalSize += Ele_MemSize;
#  endif

#  if ( FLU_SCHEME == MHM  ||  FLU_SCHEME == MHM_RP  ||  FLU_SCHEME == CTU )
   TotalSize += PriVar_MemSize + FC_Var_MemSize + FC_Flux_MemSize;

#  if ( LR_SCHEME == PPM )
   TotalSize += Slope_PPM_MemSize;
#  endif

#  ifdef MHD
   TotalSize += FC_Mag_Half_MemSize + EC_Ele_MemSize;
#  endif
#  endif // MHM/MHM_RP/CTU

#  if ( MODEL == ELBDM )
   TotalSize += Flu_MemSize_IsCompletelyRefined;
#  endif

#  if ( ELBDM_SCHEME == ELBDM_HYBRID )
   TotalSize += Flu_MemSize_HasWaveCounterpart;
#  endif

#  if ( GRAMFE_SCHEME == GRAMFE_MATMUL )
   TotalSize += GramFE_TimeEvo_MemSize;
#  endif

#  if ( MODEL != HYDRO  &&  MODEL != ELBDM )
#     warning : DO YOU WANT TO ADD SOMETHING HERE FOR THE NEW MODEL ??
#  endif

   if ( SrcTerms.Any )
   {
      TotalSize += Flu_MemSize_S_In + Flu_MemSize_S_Out;
#     ifdef MHD
      TotalSize += Mag_MemSize_S_In;
#     endif
      TotalSize += Corner_MemSize_S;
   }

#  ifdef EXACT_COOLING
   if ( SrcTerms.ExactCooling )
      TotalSize += EC_TEF_lambda_MemSize + EC_TEF_alpha_MemSize + EC_TEFc_MemSize;
#  endif

   if ( MPI_Rank == 0 )
      Aux_Message( stdout, "NOTE : total memory requirement in GPU fluid solver = %ld MB\n", TotalSize/(1<<20) );


// allocate the device memory
   DEVICE_CHECK_MALLOC(  d_Flu_Array_F_In = (real(*)[FLU_NIN][CUBE(FLU_NXT)])sycl::malloc_device( Flu_MemSize_F_In, dpct::get_in_order_queue() )  );
   DEVICE_CHECK_MALLOC(  d_Flu_Array_F_Out = (real(*)[FLU_NOUT][CUBE(PS2)])sycl::malloc_device( Flu_MemSize_F_Out, dpct::get_in_order_queue() )  );

   if ( amr->WithFlux )
   DEVICE_CHECK_MALLOC(  d_Flux_Array = (real(*)[9][NFLUX_TOTAL][SQR(PS2)])sycl::malloc_device( Flux_MemSize, dpct::get_in_order_queue() )  );

#  ifdef UNSPLIT_GRAVITY
   DEVICE_CHECK_MALLOC(  d_Pot_Array_USG_F = (real(*)[CUBE(USG_NXT_F)])sycl::malloc_device( Pot_MemSize_USG_F, dpct::get_in_order_queue() )  );

   if ( OPT__EXT_ACC )
   DEVICE_CHECK_MALLOC(  d_Corner_Array_F = (double(*)[3])sycl::malloc_device( Corner_MemSize_F, dpct::get_in_order_queue() )  );
#  endif

#  ifdef DUAL_ENERGY
   DEVICE_CHECK_MALLOC(  d_DE_Array_F_Out = (char(*)[CUBE(PS2)])sycl::malloc_device( DE_MemSize_F_Out, dpct::get_in_order_queue() )  );
#  endif

#  ifdef MHD
   DEVICE_CHECK_MALLOC(  d_Mag_Array_F_In = (real(*)[NCOMP_MAG][FLU_NXT_P1*SQR(FLU_NXT)])sycl::malloc_device( Mag_MemSize_F_In, dpct::get_in_order_queue() )  );
   DEVICE_CHECK_MALLOC(  d_Mag_Array_F_Out = (real(*)[NCOMP_MAG][PS2P1*SQR(PS2)])sycl::malloc_device( Mag_MemSize_F_Out, dpct::get_in_order_queue() )  );

   if ( amr->WithElectric )
   DEVICE_CHECK_MALLOC(  d_Ele_Array = (real(*)[9][NCOMP_ELE][PS2P1*PS2])sycl::malloc_device( Ele_MemSize, dpct::get_in_order_queue() )  );

   DEVICE_CHECK_MALLOC(  d_Mag_Array_T = (real(*)[NCOMP_MAG][PS1P1*SQR(PS1)])sycl::malloc_device( Mag_MemSize_T, dpct::get_in_order_queue() )  );
#  endif

   DEVICE_CHECK_MALLOC(  d_dt_Array_T = (real*)sycl::malloc_device( dt_MemSize_T, dpct::get_in_order_queue() )  );
   DEVICE_CHECK_MALLOC(  d_Flu_Array_T = (real(*)[FLU_NIN_T][CUBE(PS1)])sycl::malloc_device( Flu_MemSize_T, dpct::get_in_order_queue() )  );

#  if ( FLU_SCHEME == MHM  ||  FLU_SCHEME == MHM_RP  ||  FLU_SCHEME == CTU )
   DEVICE_CHECK_MALLOC(  d_FC_Var = (real(*)[6][NCOMP_TOTAL_PLUS_MAG][CUBE(N_FC_VAR)])sycl::malloc_device( FC_Var_MemSize, dpct::get_in_order_queue() )  );

   DEVICE_CHECK_MALLOC(  d_FC_Flux = (real(*)[3][NCOMP_TOTAL_PLUS_MAG][CUBE(N_FC_FLUX)])sycl::malloc_device( FC_Flux_MemSize, dpct::get_in_order_queue() )  );

   DEVICE_CHECK_MALLOC(  d_PriVar = (real(*)[NCOMP_LR][CUBE(FLU_NXT)])sycl::malloc_device( PriVar_MemSize, dpct::get_in_order_queue() )  );

#  if ( LR_SCHEME == PPM )
   DEVICE_CHECK_MALLOC(  d_Slope_PPM = (real(*)[3][NCOMP_LR][CUBE(N_SLOPE_PPM)])sycl::malloc_device( Slope_PPM_MemSize, dpct::get_in_order_queue() )  );
#  endif
#  ifdef MHD
   DEVICE_CHECK_MALLOC(  d_FC_Mag_Half = (real(*)[NCOMP_MAG][FLU_NXT_P1*SQR(FLU_NXT)])sycl::malloc_device( FC_Mag_Half_MemSize, dpct::get_in_order_queue() )  );
   DEVICE_CHECK_MALLOC(  d_EC_Ele = (real(*)[NCOMP_MAG][CUBE(N_EC_ELE)])sycl::malloc_device( EC_Ele_MemSize, dpct::get_in_order_queue() )  );
#  endif
#  endif // #if ( FLU_SCHEME == MHM  ||  FLU_SCHEME == MHM_RP  ||  FLU_SCHEME == CTU )

   if ( SrcTerms.Any ) {
   DEVICE_CHECK_MALLOC(  d_Flu_Array_S_In = (real(*)[FLU_NIN_S][CUBE(SRC_NXT)])sycl::malloc_device( Flu_MemSize_S_In, dpct::get_in_order_queue() )  );
   DEVICE_CHECK_MALLOC(  d_Flu_Array_S_Out = (real(*)[FLU_NOUT_S][CUBE(PS1)])sycl::malloc_device( Flu_MemSize_S_Out, dpct::get_in_order_queue() )  );
#  ifdef MHD
   DEVICE_CHECK_MALLOC(  d_Mag_Array_S_In = (real(*)[NCOMP_MAG][SRC_NXT_P1*SQR(SRC_NXT)])sycl::malloc_device( Mag_MemSize_S_In, dpct::get_in_order_queue() )  );
#  endif
   DEVICE_CHECK_MALLOC(  d_Corner_Array_S = (double(*)[3])sycl::malloc_device( Corner_MemSize_S, dpct::get_in_order_queue() )  );
   }


#  if ( MODEL == ELBDM )
   DEVICE_CHECK_MALLOC(  d_IsCompletelyRefined = (bool*)sycl::malloc_device( Flu_MemSize_IsCompletelyRefined, dpct::get_in_order_queue() )  );
#  endif

#  if ( ELBDM_SCHEME == ELBDM_HYBRID )
   DEVICE_CHECK_MALLOC(  d_HasWaveCounterpart = (bool(*)[CUBE(HYB_NXT)])sycl::malloc_device( Flu_MemSize_HasWaveCounterpart, dpct::get_in_order_queue() )  );
#  endif

#  if ( GRAMFE_SCHEME == GRAMFE_MATMUL )
   DEVICE_CHECK_MALLOC(  d_Flu_TimeEvo = (gramfe_matmul_float(*)[2*FLU_NXT])sycl::malloc_device( GramFE_TimeEvo_MemSize, dpct::get_in_order_queue() )  );
#  endif


#  if ( MODEL != HYDRO  &&  MODEL != ELBDM )
#     warning : DO YOU WANT TO ADD SOMETHING HERE FOR THE NEW MODEL ??
#  endif


// allocate the host memory by SYCL
   for (int t=0; t<2; t++)
   {
      DEVICE_CHECK_MALLOC(  h_Flu_Array_F_In     [t] = (real(*)[FLU_NIN][CUBE(FLU_NXT)])sycl::malloc_host( Flu_MemSize_F_In, dpct::get_in_order_queue() )  );
      DEVICE_CHECK_MALLOC(  h_Flu_Array_F_Out    [t] = (real(*)[FLU_NOUT][CUBE(PS2)])sycl::malloc_host( Flu_MemSize_F_Out, dpct::get_in_order_queue() )  );

      if ( amr->WithFlux )
      DEVICE_CHECK_MALLOC(  h_Flux_Array         [t] = (real(*)[9][NFLUX_TOTAL][SQR(PS2)])sycl::malloc_host( Flux_MemSize, dpct::get_in_order_queue() )  );

#     ifdef UNSPLIT_GRAVITY
      DEVICE_CHECK_MALLOC(  h_Pot_Array_USG_F    [t] = (real(*)[CUBE(USG_NXT_F)])sycl::malloc_host( Pot_MemSize_USG_F, dpct::get_in_order_queue() )  );

      if ( OPT__EXT_ACC )
      DEVICE_CHECK_MALLOC(  h_Corner_Array_F     [t] = (double(*)[3])sycl::malloc_host( Corner_MemSize_F, dpct::get_in_order_queue() )  );
#     endif

#     ifdef DUAL_ENERGY
      DEVICE_CHECK_MALLOC(  h_DE_Array_F_Out     [t] = (char(*)[CUBE(PS2)])sycl::malloc_host( DE_MemSize_F_Out, dpct::get_in_order_queue() )  );
#     endif

#     ifdef MHD
      DEVICE_CHECK_MALLOC(  h_Mag_Array_F_In     [t] = (real(*)[NCOMP_MAG][FLU_NXT_P1*SQR(FLU_NXT)])sycl::malloc_host( Mag_MemSize_F_In, dpct::get_in_order_queue() )  );
      DEVICE_CHECK_MALLOC(  h_Mag_Array_F_Out    [t] = (real(*)[NCOMP_MAG][PS2P1*SQR(PS2)])sycl::malloc_host( Mag_MemSize_F_Out, dpct::get_in_order_queue() )  );

      if ( amr->WithElectric )
      DEVICE_CHECK_MALLOC(  h_Ele_Array          [t] = (real(*)[9][NCOMP_ELE][PS2P1*PS2])sycl::malloc_host( Ele_MemSize, dpct::get_in_order_queue() )  );

      DEVICE_CHECK_MALLOC(  h_Mag_Array_T        [t] = (real(*)[NCOMP_MAG][PS1P1*SQR(PS1)])sycl::malloc_host( Mag_MemSize_T, dpct::get_in_order_queue() )  );
#     endif

      DEVICE_CHECK_MALLOC(  h_dt_Array_T         [t] = (real*)sycl::malloc_host( dt_MemSize_T, dpct::get_in_order_queue() )  );
      DEVICE_CHECK_MALLOC(  h_Flu_Array_T        [t] = (real(*)[FLU_NIN_T][CUBE(PS1)])sycl::malloc_host( Flu_MemSize_T, dpct::get_in_order_queue() )  );

      if ( SrcTerms.Any ) {
      DEVICE_CHECK_MALLOC(  h_Flu_Array_S_In     [t] = (real(*)[FLU_NIN_S][CUBE(SRC_NXT)])sycl::malloc_host( Flu_MemSize_S_In, dpct::get_in_order_queue() )  );
      DEVICE_CHECK_MALLOC(  h_Flu_Array_S_Out    [t] = (real(*)[FLU_NOUT_S][CUBE(PS1)])sycl::malloc_host( Flu_MemSize_S_Out, dpct::get_in_order_queue() )  );
#     ifdef MHD
      DEVICE_CHECK_MALLOC(  h_Mag_Array_S_In     [t] = (real(*)[NCOMP_MAG][SRC_NXT_P1*SQR(SRC_NXT)])sycl::malloc_host( Mag_MemSize_S_In, dpct::get_in_order_queue() )  );
#     endif
      DEVICE_CHECK_MALLOC(  h_Corner_Array_S     [t] = (double(*)[3])sycl::malloc_host( Corner_MemSize_S, dpct::get_in_order_queue() )  );
      }

#     if ( MODEL == ELBDM )
      DEVICE_CHECK_MALLOC(  h_IsCompletelyRefined[t] = (bool*)sycl::malloc_host( Flu_MemSize_IsCompletelyRefined, dpct::get_in_order_queue() )  );
#     endif

#     if ( ELBDM_SCHEME == ELBDM_HYBRID )
      DEVICE_CHECK_MALLOC(  h_HasWaveCounterpart [t] = (bool(*)[CUBE(HYB_NXT)])sycl::malloc_host( Flu_MemSize_HasWaveCounterpart, dpct::get_in_order_queue() )  );
#     endif
   } // for (int t=0; t<2; t++)

#  if ( GRAMFE_SCHEME == GRAMFE_MATMUL )
   DEVICE_CHECK_MALLOC(  h_GramFE_TimeEvo = (gramfe_matmul_float(*)[2*FLU_NXT])sycl::malloc_host( GramFE_TimeEvo_MemSize, dpct::get_in_order_queue() )  );
#  endif

// create streams
   Stream = new dpct::queue_ptr [GPU_NStream];
   for (int s=0; s<GPU_NStream; s++)      DEVICE_CHECK_ERROR(  DPCT_CHECK_ERROR( Stream[s] = dpct::get_current_device().create_queue() )  );


   return GAMER_SUCCESS;

} // FUNCTION : GPU_MemAllocate_Fluid



#endif // #ifdef GPU
