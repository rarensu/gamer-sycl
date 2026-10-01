#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GPUAPI.h"

#if ( defined GPU  &&  defined GRAVITY )



extern real (*d_Rho_Array_P    )[ CUBE(RHO_NXT) ];
extern real (*d_Pot_Array_P_In )[ CUBE(POT_NXT) ];
extern real (*d_Pot_Array_P_Out)[ CUBE(GRA_NXT) ];
#ifdef UNSPLIT_GRAVITY
extern real (*d_Pot_Array_USG_G)[ CUBE(USG_NXT_G) ];
extern real (*d_Flu_Array_USG_G)[GRA_NIN-1][ CUBE(PS1) ];
#endif
extern real (*d_Flu_Array_G    )[GRA_NIN  ][ CUBE(PS1) ];
extern double (*d_Corner_Array_PGT)[3];
#ifdef DUAL_ENERGY
extern char (*d_DE_Array_G     )[ CUBE(PS1) ];
#endif
#ifdef MHD
extern real (*d_Emag_Array_G   )[ CUBE(PS1) ];
#endif
extern real (*d_Pot_Array_T)    [ CUBE(GRA_NXT) ];
extern real  *d_ExtPotTable;
extern void **d_ExtPotGenePtr;




//-------------------------------------------------------------------------------------------------------
// Function    :  GPU_MemAllocate_PoissonGravity
// Description :  Allocate device and host memory for the Poisson and Gravity solvers
//
// Parameter   :  Pot_NPG  : Number of patch groups evaluated simultaneously by GPU
//
// Return      :  GAMER_SUCCESS / GAMER_FAILED
//-------------------------------------------------------------------------------------------------------
int GPU_MemAllocate_PoissonGravity( const int Pot_NPG )
{

   const long Pot_NP            = 8*Pot_NPG;
   const long Rho_MemSize_P     = sizeof(real  )*Pot_NP*CUBE(RHO_NXT);
   const long Pot_MemSize_P_In  = sizeof(real  )*Pot_NP*CUBE(POT_NXT);
   const long Pot_MemSize_P_Out = sizeof(real  )*Pot_NP*CUBE(GRA_NXT);
#  ifdef UNSPLIT_GRAVITY
   const long Pot_MemSize_USG_G = sizeof(real  )*Pot_NP*CUBE(USG_NXT_G);
   const long Flu_MemSize_USG_G = sizeof(real  )*Pot_NP*CUBE(PS1)*(GRA_NIN-1);
#  endif
   const long Flu_MemSize_G     = sizeof(real  )*Pot_NP*CUBE(PS1)*(GRA_NIN  );
   const long Corner_MemSize    = sizeof(double)*Pot_NP*3;
#  ifdef DUAL_ENERGY
   const long DE_MemSize_G      = sizeof(char  )*Pot_NP*CUBE(PS1);
#  endif
#  ifdef MHD
   const long Emag_MemSize_G    = sizeof(real  )*Pot_NP*CUBE(PS1);
#  endif
   const long Pot_MemSize_T     = sizeof(real  )*Pot_NP*CUBE(GRA_NXT);
   const long ExtPot_MemSize    = (long)sizeof(real)*EXT_POT_TABLE_NPOINT[0]*EXT_POT_TABLE_NPOINT[1]*EXT_POT_TABLE_NPOINT[2];
   const long GenePtr_MemSize   = sizeof(void* )*EXT_POT_NGENE_MAX;


// output the total memory requirement
   long TotalSize = Rho_MemSize_P + Pot_MemSize_P_In + Pot_MemSize_P_Out + Flu_MemSize_G + Pot_MemSize_T;
#  ifdef UNSPLIT_GRAVITY
   TotalSize += Pot_MemSize_USG_G + Flu_MemSize_USG_G;
#  endif
   if ( OPT__EXT_ACC  ||  OPT__EXT_POT )
   TotalSize += Corner_MemSize;
#  ifdef DUAL_ENERGY
   TotalSize += DE_MemSize_G;
#  endif
#  ifdef MHD
   TotalSize += Emag_MemSize_G;
#  endif
   TotalSize += GenePtr_MemSize;
   if ( OPT__EXT_POT == EXT_POT_TABLE )
   TotalSize += ExtPot_MemSize;

   if ( MPI_Rank == 0 )
      Aux_Message( stdout, "NOTE : total memory requirement in GPU Poisson and gravity solver = %ld MB\n",
                   TotalSize/(1<<20) );


// allocate the device memory
   DEVICE_CHECK_MALLOC( d_Rho_Array_P = (real(*)[CUBE(RHO_NXT)])sycl::malloc_device(
                           Rho_MemSize_P, dpct::get_in_order_queue()) );
   DEVICE_CHECK_MALLOC( d_Pot_Array_P_In = (real(*)[CUBE(POT_NXT)])sycl::malloc_device(
                           Pot_MemSize_P_In, dpct::get_in_order_queue()) );
   DEVICE_CHECK_MALLOC( d_Pot_Array_P_Out = (real(*)[CUBE(GRA_NXT)])sycl::malloc_device(
                           Pot_MemSize_P_Out, dpct::get_in_order_queue()) );
#  ifdef UNSPLIT_GRAVITY
   DEVICE_CHECK_MALLOC( d_Pot_Array_USG_G = (real(*)[CUBE(USG_NXT_G)])sycl::malloc_device(
                           Pot_MemSize_USG_G, dpct::get_in_order_queue()) );
   DEVICE_CHECK_MALLOC( d_Flu_Array_USG_G = (real(*)[GRA_NIN-1][CUBE(PS1)])sycl::malloc_device(
                           Flu_MemSize_USG_G, dpct::get_in_order_queue()) );
#  endif
   DEVICE_CHECK_MALLOC( d_Flu_Array_G = (real(*)[GRA_NIN][CUBE(PS1)])sycl::malloc_device(
                           Flu_MemSize_G, dpct::get_in_order_queue()) );

   if ( OPT__EXT_ACC  ||  OPT__EXT_POT )
   DEVICE_CHECK_MALLOC( d_Corner_Array_PGT = (double(*)[3])sycl::malloc_device(
                           Corner_MemSize, dpct::get_in_order_queue()) );

#  ifdef DUAL_ENERGY
   DEVICE_CHECK_MALLOC( d_DE_Array_G = (char(*)[CUBE(PS1)])sycl::malloc_device(
                           DE_MemSize_G, dpct::get_in_order_queue()) );
#  endif

#  ifdef MHD
   DEVICE_CHECK_MALLOC( d_Emag_Array_G = (real(*)[CUBE(PS1)])sycl::malloc_device(
                           Emag_MemSize_G, dpct::get_in_order_queue()) );
#  endif

   DEVICE_CHECK_MALLOC( d_Pot_Array_T = (real(*)[CUBE(GRA_NXT)])sycl::malloc_device(
                           Pot_MemSize_T, dpct::get_in_order_queue()) );

   if ( OPT__EXT_POT == EXT_POT_TABLE )
   DEVICE_CHECK_MALLOC( d_ExtPotTable = (real *)sycl::malloc_device(
                           ExtPot_MemSize, dpct::get_in_order_queue()) );

   DEVICE_CHECK_MALLOC( d_ExtPotGenePtr = (void **)sycl::malloc_device(
                           GenePtr_MemSize, dpct::get_in_order_queue()) );


// allocate the host memory by SYCL
   for (int t=0; t<2; t++)
   {
      DEVICE_CHECK_MALLOC( h_Rho_Array_P[t] = (real(*)[CUBE(RHO_NXT)])sycl::malloc_host(
                              Rho_MemSize_P, dpct::get_in_order_queue()) );
      DEVICE_CHECK_MALLOC( h_Pot_Array_P_In[t] = (real(*)[CUBE(POT_NXT)])sycl::malloc_host(
                              Pot_MemSize_P_In, dpct::get_in_order_queue()) );
      DEVICE_CHECK_MALLOC( h_Pot_Array_P_Out[t] = (real(*)[CUBE(GRA_NXT)])sycl::malloc_host(
                              Pot_MemSize_P_Out, dpct::get_in_order_queue()) );
#     ifdef UNSPLIT_GRAVITY
      DEVICE_CHECK_MALLOC( h_Pot_Array_USG_G[t] = (real(*)[CUBE(USG_NXT_G)])sycl::malloc_host(
                              Pot_MemSize_USG_G, dpct::get_in_order_queue()) );
      DEVICE_CHECK_MALLOC( h_Flu_Array_USG_G[t] = (real(*)[GRA_NIN-1][CUBE(PS1)])sycl::malloc_host(
                              Flu_MemSize_USG_G, dpct::get_in_order_queue()) );
#     endif
      DEVICE_CHECK_MALLOC( h_Flu_Array_G[t] = (real(*)[GRA_NIN][CUBE(PS1)])sycl::malloc_host(
                              Flu_MemSize_G, dpct::get_in_order_queue()) );

      if ( OPT__EXT_ACC  ||  OPT__EXT_POT )
      DEVICE_CHECK_MALLOC( h_Corner_Array_PGT[t] = (double(*)[3])sycl::malloc_host(
                              Corner_MemSize, dpct::get_in_order_queue()) );

#     ifdef DUAL_ENERGY
      DEVICE_CHECK_MALLOC( h_DE_Array_G[t] = (char(*)[CUBE(PS1)])sycl::malloc_host(
                              DE_MemSize_G, dpct::get_in_order_queue()) );
#     endif

#     ifdef MHD
      DEVICE_CHECK_MALLOC( h_Emag_Array_G[t] = (real(*)[CUBE(PS1)])sycl::malloc_host(
                              Emag_MemSize_G, dpct::get_in_order_queue()) );
#     endif

      DEVICE_CHECK_MALLOC( h_Pot_Array_T[t] = (real(*)[CUBE(GRA_NXT)])sycl::malloc_host(
                              Pot_MemSize_T, dpct::get_in_order_queue()) );
   } // for (int t=0; t<2; t++)

   if ( OPT__EXT_POT == EXT_POT_TABLE )
      DEVICE_CHECK_MALLOC( h_ExtPotTable = (real *)sycl::malloc_host(
                              ExtPot_MemSize, dpct::get_in_order_queue()) );

      DEVICE_CHECK_MALLOC( h_ExtPotGenePtr = (void **)sycl::malloc_host(
                              GenePtr_MemSize, dpct::get_in_order_queue()) );


   return GAMER_SUCCESS;

} // FUNCTION : GPU_MemAllocate_PoissonGravity



#endif // #if ( defined GPU  &&  defined GRAVITY )
