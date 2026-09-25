#define DPCT_COMPAT_RT_VERSION 12090
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GPUAPI.h"

void Aux_GetCPUInfo( const char *FileName );

#ifdef GPU




//-------------------------------------------------------------------------------------------------------
// Function    :  GPU_DiagnoseDevice
// Description :  Take a diagnosis of each GPU
//-------------------------------------------------------------------------------------------------------
void GPU_DiagnoseDevice()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// get the hostname and PID of each process
   const int PID = getpid();
   char Host[1024];
   gethostname( Host, 1024 );


// get the number of devices
   int DeviceCount;
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(DeviceCount = dpct::device_count()));

   if ( DeviceCount == 0 )
      Aux_Error( ERROR_INFO, "no devices supporting CUDA at MPI_Rank %2d (host = %8s) !!\n", MPI_Rank, Host );


// get the device ID
   int GetDeviceID = 999;
   DEVICE_CHECK_ERROR(
       DPCT_CHECK_ERROR(GetDeviceID = dpct::get_current_device_id()));

// load the device properties
   dpct::device_info DeviceProp;
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_device(GetDeviceID).get_device_info(DeviceProp)));

// get the number of cores per multiprocessor
   int NCorePerMP;
   /*
   DPCT1005:44: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   if (DeviceProp.get_major_version() == 2 &&
       DeviceProp.get_minor_version() == 0) NCorePerMP = 32;
   /*
   DPCT1005:45: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 2 &&
            DeviceProp.get_minor_version() == 1) NCorePerMP = 48;
   /*
   DPCT1005:46: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 3) NCorePerMP = 192;
   /*
   DPCT1005:47: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 5) NCorePerMP = 128;
   /*
   DPCT1005:48: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 6) NCorePerMP = 64;
   /*
   DPCT1005:49: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 7) NCorePerMP = 64;
   /*
   DPCT1005:50: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 8 &&
            DeviceProp.get_minor_version() == 0) NCorePerMP = 64;
   /*
   DPCT1005:51: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 8 &&
            DeviceProp.get_minor_version() == 6) NCorePerMP = 128;
   /*
   DPCT1005:52: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 8 &&
            DeviceProp.get_minor_version() == 9) NCorePerMP = 128;
   /*
   DPCT1005:53: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 9) NCorePerMP = 128;
   /*
   DPCT1005:54: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 10 &&
            DeviceProp.get_minor_version() == 0) NCorePerMP = 128;
   /*
   DPCT1005:55: The SYCL device version is different from CUDA Compute
   Compatibility. You may need to rewrite this code.
   */
   else if (DeviceProp.get_major_version() == 12 &&
            DeviceProp.get_minor_version() == 0) NCorePerMP = 128;
   else
      fprintf(stderr,
              "WARNING : unable to determine the number of cores per "
              "multiprocessor for version %d.%d ...\n",
              /*
              DPCT1005:56: The SYCL device version is different from CUDA
              Compute Compatibility. You may need to rewrite this code.
              */
              DeviceProp.get_major_version(), DeviceProp.get_minor_version());

// record the device properties
   char FileName[2*MAX_STRING];
   sprintf( FileName, "%s/Record__Note", OUTPUT_DIR );

   if ( MPI_Rank == 0 )
   {
       FILE *Note = fopen( FileName, "a" );
       fprintf( Note, "Device Diagnosis\n" );
       fprintf( Note, "***********************************************************************************\n" );
       fclose( Note );
   }

   for (int YourTurn=0; YourTurn<MPI_NRank; YourTurn++)
   {
      if ( MPI_Rank == YourTurn )
      {
         int DriverVersion = 0, RuntimeVersion = 0;
         /*
         DPCT1043:57: The version-related API is different in SYCL. An initial
         code was generated, but you need to adjust it.
         */
         DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
             DriverVersion =
                 dpct::get_major_version(dpct::get_current_device())));
         /*
         DPCT1043:58: The version-related API is different in SYCL. An initial
         code was generated, but you need to adjust it.
         */
         DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
             RuntimeVersion =
                 dpct::get_major_version(dpct::get_current_device())));

         FILE *Note = fopen( FileName, "a" );
         if ( MPI_Rank != 0 )   fprintf( Note, "\n\n" );
         fprintf( Note, "MPI_Rank = %3d, hostname = %10s, PID = %5d\n\n", MPI_Rank, Host, PID );
         fprintf( Note, "CPU Info :\n" );
         fflush( Note );

         Aux_GetCPUInfo( FileName );

         int clockRate; // in unit of kHz
         DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
             clockRate =
                 dpct::get_device(GetDeviceID).get_max_clock_frequency()));

         fprintf( Note, "\n" );
         fprintf( Note, "GPU Info :\n" );
         fprintf( Note, "Number of GPUs                          : %d\n"     , DeviceCount );
         fprintf( Note, "GPU ID                                  : %d\n"     , GetDeviceID );
         fprintf(Note, "GPU Name                                : %s\n",
                 DeviceProp.get_name());
         fprintf( Note, "CUDA Driver Version                     : %d.%d\n"  , DriverVersion/1000, DriverVersion%100 );
         fprintf( Note, "CUDA Runtime Version                    : %d.%d\n"  , RuntimeVersion/1000, RuntimeVersion%100 );
         /*
         DPCT1005:59: The SYCL device version is different from CUDA Compute
         Compatibility. You may need to rewrite this code.
         */
         fprintf(Note, "CUDA Major Revision Number              : %d\n",
                 DeviceProp.get_major_version());
         /*
         DPCT1005:60: The SYCL device version is different from CUDA Compute
         Compatibility. You may need to rewrite this code.
         */
         fprintf(Note, "CUDA Minor Revision Number              : %d\n",
                 DeviceProp.get_minor_version());
         fprintf( Note, "Clock Rate                              : %f GHz\n" , clockRate/1.0e6 );
         fprintf(Note, "Global Memory Size                      : %ld MB\n",
                 (long)DeviceProp.get_global_mem_size() / 1024 / 1024);
         /*
         DPCT1051:61: SYCL does not support a device property functionally
         compatible with totalConstMem. It was migrated to get_global_mem_size.
         You may need to adjust the value of get_global_mem_size for the
         specific device.
         */
         fprintf(Note, "Constant Memory Size                    : %ld KB\n",
                 (long)DeviceProp.get_global_mem_size() / 1024);
         /*
         DPCT1019:62: local_mem_size in SYCL is not a complete equivalent of
         sharedMemPerBlock in CUDA. You may need to adjust the code.
         */
         fprintf(Note, "Shared Memory Size per Block            : %ld KB\n",
                 (long)DeviceProp.get_local_mem_size() / 1024);
         /*
         DPCT1051:63: SYCL does not support a device property functionally
         compatible with regsPerBlock. It was migrated to
         get_max_register_size_per_work_group. You may need to adjust the value
         of get_max_register_size_per_work_group for the specific device.
         */
         fprintf(Note, "Number of Registers per Block           : %d\n",
                 DeviceProp.get_max_register_size_per_work_group());
         fprintf(Note, "Warp Size                               : %d\n",
                 DeviceProp.get_max_sub_group_size());
         fprintf(Note, "Number of Multiprocessors:              : %d\n",
                 DeviceProp.get_max_compute_units());
         fprintf( Note, "Number of FP32 Cores per Multiprocessor : %d\n"     , NCorePerMP );
         fprintf(Note, "Total Number of Cores:                  : %d\n",
                 DeviceProp.get_max_compute_units() * NCorePerMP);
         fprintf(Note, "Max Number of Threads per Block         : %d\n",
                 DeviceProp.get_max_work_group_size());
         fprintf(Note, "Max Size of the Block X-Dimension       : %d\n",
                 DeviceProp.get_max_work_item_sizes<int *>()[0]);
         /*
         DPCT1022:64: There is no exact match between the maxGridSize and the
         max_nd_range size. Verify the correctness of the code.
         */
         fprintf(Note, "Max Size of the Grid X-Dimension        : %d\n",
                 DeviceProp.get_max_nd_range_size<int *>()[0]);
         fprintf( Note, "Concurrent Copy and Execution           : %s\n"     , DeviceProp.asyncEngineCount>0  ? "Yes" : "No" );
         fprintf( Note, "Concurrent Up/Downstream Copies         : %s\n"     , DeviceProp.asyncEngineCount==2 ? "Yes" : "No" );
#if (DPCT_COMPAT_RT_VERSION >= 3000)
         /*
         DPCT1051:65: SYCL does not support a device property functionally
         compatible with concurrentKernels. It was migrated to true. You may
         need to adjust the value of true for the specific device.
         */
         fprintf(Note, "Concurrent Kernel Execution             : %s\n",
                 true ? "Yes" : "No");
#        endif
#if (DPCT_COMPAT_RT_VERSION >= 3010)
         fprintf(
             Note, "GPU has ECC Support Enabled             : %s\n",
             dpct::get_current_device()
                     .get_info<sycl::info::device::error_correction_support>()
                 ? "Yes"
                 : "No");
#        endif

         fclose( Note );
       }

       MPI_Barrier( MPI_COMM_WORLD );

   } // for (int YourTurn=0; YourTurn<NGPU; YourTurn++)

   if ( MPI_Rank == 0 )
   {
      FILE *Note = fopen( FileName, "a" );
      fprintf( Note, "***********************************************************************************\n" );
      fclose( Note );

      Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );
   }

} // FUNCTION : GPU_DiagnoseDevice



#endif // #ifdef GPU
