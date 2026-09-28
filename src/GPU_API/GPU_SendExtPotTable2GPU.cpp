#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GPUAPI.h"
#include "POT.h"

#if ( defined GPU  &&  defined GRAVITY )


// device pointer
extern real *d_ExtPotTable;




//-------------------------------------------------------------------------------------------------------
// Function    :  GPU_SendExtPotTable2GPU
// Description :  Send the external potential table to GPU
//
// Note        :  1. Invoked by Init_LoadExtPotTable()
//                2. Use synchronous transfer
//
// Parameter   :  h_Table : Host array storing the input table
//
// Return      :  d_ExtPotTable
//-------------------------------------------------------------------------------------------------------
void GPU_SendExtPotTable2GPU( const real *h_Table )
{

   const long MemSize = (long)sizeof(real)*EXT_POT_TABLE_NPOINT[0]*EXT_POT_TABLE_NPOINT[1]*EXT_POT_TABLE_NPOINT[2];

// use synchronous transfer
   DEVICE_CHECK_ERROR(DPCT_CHECK_ERROR(
       dpct::get_in_order_queue()
           .memcpy(d_ExtPotTable, h_Table, MemSize)
           .wait()));

} // FUNCTION : GPU_SendExtPotTable2GPU



#endif // #if ( defined GPU  &&  defined GRAVITY )
