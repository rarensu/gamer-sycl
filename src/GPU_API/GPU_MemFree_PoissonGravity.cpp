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
extern real (*d_Pot_Array_T    )[ CUBE(GRA_NXT) ];
extern real  *d_ExtPotTable;
extern void **d_ExtPotGenePtr;




//-------------------------------------------------------------------------------------------------------
// Function    :  GPU_MemFree_PoissonGravity
// Description :  Free the device and host memory previously allocated by GPU_MemAllocate_PoissonGravity()
//
// Parameter   :  None
//-------------------------------------------------------------------------------------------------------
void GPU_MemFree_PoissonGravity()
{

// free the device memory
   if ( d_Rho_Array_P      != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Rho_Array_P,      dpct::get_in_order_queue() )  ));  d_Rho_Array_P      = NULL; }
   if ( d_Pot_Array_P_In   != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Pot_Array_P_In,   dpct::get_in_order_queue() )  ));  d_Pot_Array_P_In   = NULL; }
   if ( d_Pot_Array_P_Out  != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Pot_Array_P_Out,  dpct::get_in_order_queue() )  ));  d_Pot_Array_P_Out  = NULL; }
#  ifdef UNSPLIT_GRAVITY
   if ( d_Pot_Array_USG_G  != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Pot_Array_USG_G,  dpct::get_in_order_queue() )  ));  d_Pot_Array_USG_G  = NULL; }
   if ( d_Flu_Array_USG_G  != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Flu_Array_USG_G,  dpct::get_in_order_queue() )  ));  d_Flu_Array_USG_G  = NULL; }
#  endif
   if ( d_Flu_Array_G      != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Flu_Array_G,      dpct::get_in_order_queue() )  ));  d_Flu_Array_G      = NULL; }
   if ( d_Corner_Array_PGT != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Corner_Array_PGT, dpct::get_in_order_queue() )  ));  d_Corner_Array_PGT = NULL; }
#  ifdef DUAL_ENERGY
   if ( d_DE_Array_G       != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_DE_Array_G,       dpct::get_in_order_queue() )  ));  d_DE_Array_G       = NULL; }
#  endif
#  ifdef MHD
   if ( d_Emag_Array_G     != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Emag_Array_G,     dpct::get_in_order_queue() )  ));  d_Emag_Array_G     = NULL; }
#  endif
   if ( d_Pot_Array_T      != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_Pot_Array_T,      dpct::get_in_order_queue() )  ));  d_Pot_Array_T      = NULL; }
   if ( d_ExtPotTable      != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_ExtPotTable,      dpct::get_in_order_queue() )  ));  d_ExtPotTable      = NULL; }
   if ( d_ExtPotGenePtr    != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  dpct::dpct_free( d_ExtPotGenePtr,    dpct::get_in_order_queue() )  ));  d_ExtPotGenePtr    = NULL; }


// free the host memory allocated by SYCL
   for (int t=0; t<2; t++)
   {
      if ( h_Rho_Array_P     [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Rho_Array_P     [t], dpct::get_in_order_queue() )  ));  h_Rho_Array_P     [t] = NULL; }
      if ( h_Pot_Array_P_In  [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Pot_Array_P_In  [t], dpct::get_in_order_queue() )  ));  h_Pot_Array_P_In  [t] = NULL; }
      if ( h_Pot_Array_P_Out [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Pot_Array_P_Out [t], dpct::get_in_order_queue() )  ));  h_Pot_Array_P_Out [t] = NULL; }
#     ifdef UNSPLIT_GRAVITY
      if ( h_Pot_Array_USG_G [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Pot_Array_USG_G [t], dpct::get_in_order_queue() )  ));  h_Pot_Array_USG_G [t] = NULL; }
      if ( h_Flu_Array_USG_G [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Flu_Array_USG_G [t], dpct::get_in_order_queue() )  ));  h_Flu_Array_USG_G [t] = NULL; }
#     endif
      if ( h_Flu_Array_G     [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Flu_Array_G     [t], dpct::get_in_order_queue() )  ));  h_Flu_Array_G     [t] = NULL; }
      if ( h_Corner_Array_PGT[t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Corner_Array_PGT[t], dpct::get_in_order_queue() )  ));  h_Corner_Array_PGT[t] = NULL; }
#     ifdef DUAL_ENERGY
      if ( h_DE_Array_G      [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_DE_Array_G      [t], dpct::get_in_order_queue() )  ));  h_DE_Array_G      [t] = NULL; }
#     endif
#     ifdef MHD
      if ( h_Emag_Array_G    [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Emag_Array_G    [t], dpct::get_in_order_queue() )  ));  h_Emag_Array_G    [t] = NULL; }
#     endif
      if ( h_Pot_Array_T     [t] != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_Pot_Array_T     [t], dpct::get_in_order_queue() )  ));  h_Pot_Array_T     [t] = NULL; }
   } // for (int t=0; t<2; t++)

      if ( h_ExtPotTable         != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_ExtPotTable,         dpct::get_in_order_queue() )  ));  h_ExtPotTable         = NULL; }
      if ( h_ExtPotGenePtr       != NULL ) {  DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(  sycl::free( h_ExtPotGenePtr,       dpct::get_in_order_queue() )  ));  h_ExtPotGenePtr       = NULL; }

} // FUNCTION : GPU_MemFree_PoissonGravity



#endif // #if ( defined GPU  &&  defined GRAVITY )
