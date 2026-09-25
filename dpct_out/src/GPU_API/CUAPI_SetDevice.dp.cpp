#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUAPI.h"
#include "FLU.h"
#ifdef GRAVITY
#include "CUPOT.h"
#endif
#ifdef LAOHU
extern "C" { int GetFreeGpuDevID( int, int ); }
#endif

#ifdef GPU




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_SetDevice
// Description :  Set the active device
//
// Parameter   :  Mode :    -3 --> set by the gpudevmgr library on the NAOC Laohu cluster
//                          -2 --> set automatically by CUDA (must work with the "compute-exclusive mode")
//                          -1 --> set by MPI ranks : SetDeviceID = MPI_Rank % DeviceCount
//                       >=  0 --> set to "Mode"
//-------------------------------------------------------------------------------------------------------
void CUAPI_SetDevice( const int Mode )
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// check
#  ifdef LAOHU
   if ( Mode < -3 )     Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "Mode", Mode );
   if ( Mode != -3  &&  MPI_Rank == 0 )
      Aux_Message( stderr, "WARNING : \"OPT__GPUID_SELECT != -3\" on the Laohu cluster !?\n" );
#  else
   if ( Mode < -2 )     Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "Mode", Mode );
#  endif


// get the hostname of each MPI process
   char Host[1024];
   gethostname( Host, 1024 );


// verify that there are GPU supporting CUDA
   int DeviceCount;
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(DeviceCount = dpct::device_count()));

   if ( DeviceCount == 0 )
      Aux_Error( ERROR_INFO, "no devices support CUDA at MPI_Rank %2d (host = %8s) !!\n", MPI_Rank, Host );


// set the device ID
   void **d_TempPtr = NULL;
   int SetDeviceID, GetDeviceID = 999;
   int computeMode;
   dpct::device_info DeviceProp;

   switch ( Mode )
   {
#     ifdef LAOHU
      case -3:
         SetDeviceID = GetFreeGpuDevID( DeviceCount, MPI_Rank );

         if ( SetDeviceID < DeviceCount )
            CUDA_CHECK_ERROR(  cudaSetDevice( SetDeviceID )  );

         else
            Aux_Error( ERROR_INFO, "SetDeviceID (%d) >= DeviceCount (%d) at MPI_Rank %2d (host = %8s) !!\n",
                       SetDeviceID, DeviceCount, MPI_Rank, Host );
         break;
#     endif


      case -2:
         CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
             d_TempPtr = (void **)sycl::malloc_device(
                 sizeof(int),
                 dpct::get_in_order_queue()))); // to set the GPU ID
         CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
             dpct::dpct_free(d_TempPtr, dpct::get_in_order_queue())));

//       make sure that the "exclusive" compute mode is adopted
         CUDA_CHECK_ERROR(
             DPCT_CHECK_ERROR(GetDeviceID = dpct::get_current_device_id()));
         /*
         DPCT1035:79: All SYCL devices can be used by the host to submit tasks.
         You may need to adjust this code.
         */
         CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(computeMode = 1));

         /*
         DPCT1035:80: All SYCL devices can be used by the host to submit tasks.
         You may need to adjust this code.
         */
         if (computeMode != 0)
         {
            Aux_Message( stderr, "WARNING : \"exclusive\" compute mode is NOT enabled for \"%s\" at Rank %2d",
                         "OPT__GPUID_SELECT == -2", MPI_Rank );
            Aux_Message( stderr, " (host=%8s) !!\n", Host );
         }
         break;


      case -1:
         SetDeviceID = MPI_Rank % DeviceCount;
         /*
         DPCT1093:81: The "SetDeviceID" device may be not the one intended for
         use. Adjust the selected device if needed.
         */
         CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(dpct::select_device(SetDeviceID)));

         if ( MPI_NRank > 1  &&  MPI_Rank == 0 )
         {
            Aux_Message( stderr, "WARNING : please make sure that different MPI ranks will use different GPUs " );
            Aux_Message( stderr, "for \"%s\" !!\n", "OPT__GPUID_SELECT == -1" );
         }
         break;


      default:
         SetDeviceID = Mode;

         if ( SetDeviceID < DeviceCount )
            /*
            DPCT1093:82: The "SetDeviceID" device may be not the one intended
            for use. Adjust the selected device if needed.
            */
            CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(dpct::select_device(SetDeviceID)));

         else
            Aux_Error( ERROR_INFO, "SetDeviceID (%d) >= DeviceCount (%d) at MPI_Rank %2d (host = %8s) !!\n",
                       SetDeviceID, DeviceCount, MPI_Rank, Host );

         if ( MPI_NRank > 1  &&  MPI_Rank == 0 )
         {
            Aux_Message( stderr, "WARNING : please make sure that different MPI ranks will use different GPUs " );
            Aux_Message( stderr, "for \"%s\" !!\n", "OPT__GPUID_SELECT == -1" );
         }
         break;
   } // switch ( Mode )


// check
// (0) load the device properties and the versions of CUDA and driver
   int DriverVersion = 0, RuntimeVersion = 0;
   CUDA_CHECK_ERROR(
       DPCT_CHECK_ERROR(GetDeviceID = dpct::get_current_device_id()));
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_device(GetDeviceID).get_device_info(DeviceProp)));
   /*
   DPCT1043:83: The version-related API is different in SYCL. An initial code
   was generated, but you need to adjust it.
   */
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       DriverVersion = dpct::get_major_version(dpct::get_current_device())));
   /*
   DPCT1043:84: The version-related API is different in SYCL. An initial code
   was generated, but you need to adjust it.
   */
   CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
       RuntimeVersion = dpct::get_major_version(dpct::get_current_device())));

// (1) verify the device version
   /*
   DPCT1005:85: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   if (DeviceProp.get_major_version() < 1)
      Aux_Error( ERROR_INFO, "\ndevice major version < 1 at MPI_Rank %2d (host = %8s) !!\n", MPI_Rank, Host );

   if ( Mode >= -1 )
   {
//    (2) verify that the device ID is properly set
      if ( GetDeviceID != SetDeviceID )
         Aux_Error( ERROR_INFO, "GetDeviceID (%d) != SetDeviceID (%d) at MPI_Rank %2d (host = %8s) !!\n",
                    GetDeviceID, SetDeviceID, MPI_Rank, Host );

//    (3) verify that the adopted ID is accessible
      CUDA_CHECK_ERROR(
          DPCT_CHECK_ERROR(d_TempPtr = (void **)sycl::malloc_device(
                               sizeof(int), dpct::get_in_order_queue())));
      CUDA_CHECK_ERROR(DPCT_CHECK_ERROR(
          dpct::dpct_free(d_TempPtr, dpct::get_in_order_queue())));
   }


// (4) verify the capability of double precision
#  ifdef FLOAT8
   if ( DeviceProp.major < 2  &&  DeviceProp.minor < 3 )
      Aux_Error( ERROR_INFO, "GPU \"%s\" at MPI_Rank %2d (host = %8s) does not support FLOAT8 !!\n",
                 DeviceProp.name, MPI_Rank, Host );
#  endif


// (5) verify the GPU compute capability
   /*
   DPCT1005:86: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   if (DeviceProp.get_major_version() * 100 +
           DeviceProp.get_minor_version() * 10 !=
       GPU_COMPUTE_CAPABILITY)
      Aux_Error(
          ERROR_INFO,
          "The compute capability %d.%d of the GPU \"%s\" does not match the "
          "GPU_COMPUTE_CAPABILITY %d !!\n"
          "        --> Please set it properly in your machine config file.\n",
          /*
          DPCT1005:87: The SYCL device version is different from CUDA Compute
          Compatibility. You may need to rewrite this code.
          */
          DeviceProp.get_major_version(), DeviceProp.get_minor_version(),
          DeviceProp.get_name(), GPU_COMPUTE_CAPABILITY);

// (6) some options are not supported
// (6-1) fluid solver
#  if ( MODEL == HYDRO )
#  if (  defined FLOAT8  &&  CHECK_INTERMEDIATE == EXACT  && \
         ( FLU_SCHEME == MHM || FLU_SCHEME == MHM_RP || FLU_SCHEME == CTU )  )
      if ( RuntimeVersion < 3020 )
         Aux_Error( ERROR_INFO, "CHECK_INTERMEDIATE == EXACT + FLOAT8 is not supported in CUDA < 3.2 !!" );
#  endif
#  endif // #if ( MODEL == HYDRO )

// (6-2) SOR Poisson solver
#  if ( POT_SCHEME == SOR )
#     ifdef SOR_USE_SHUFFLE
      if ( DeviceProp.warpSize != 32 )
         Aux_Error( ERROR_INFO, "warp size (%d) != 32 !!\n", DeviceProp.warpSize );

      if ( DeviceProp.maxThreadsPerBlock > 1024 )
         Aux_Error( ERROR_INFO, "maximum number of threads per block (%d) > 1024 !!\n", DeviceProp.maxThreadsPerBlock );
#     endif

#     ifdef SOR_USE_PADDING
      if ( DeviceProp.warpSize != 32 )
         Aux_Error( ERROR_INFO, "warp size (%d) != 32 !!\n", DeviceProp.warpSize );

      if ( POT_GHOST_SIZE != 5 )
         Aux_Error( ERROR_INFO, "POT_GHOST_SIZE (%d) != 5 !!\n", POT_GHOST_SIZE );
#     endif
#  endif // if ( POT_SCHEME == SOR )


// (7) warp size
   if (DeviceProp.get_max_sub_group_size() != WARP_SIZE)
      Aux_Error(ERROR_INFO,
                "inconsistent warp size (warpSize %d, WARP_SIZE %d) !!\n",
                DeviceProp.get_max_sub_group_size(), WARP_SIZE);

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : CUAPI_SetDevice



#endif // #ifdef GPU
