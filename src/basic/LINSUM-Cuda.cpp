//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LINSUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define LINSUM_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(y, m_y, iend);

#define LINSUM_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, y, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y);

/* __global__ void linsum(Real_ptr y, Real_ptr x, */ 
/*                       Real_type a, */ 
/*                       Index_type iend) */ 
/* { */
/*    Index_type i = blockIdx.x * blockDim.x + threadIdx.x; */
/*    if (i < iend) { */
/*      LINSUM_BODY; */ 
/*    } */
/* } */


void LINSUM::runCudaVariant(VariantID vid)
{
  /* const Index_type run_reps = getRunReps(); */
  /* const Index_type ibegin = 0; */
  /* const Index_type iend = getRunSize(); */

  /* LINSUM_DATA_SETUP; */

  /* if ( vid == Base_CUDA ) { */

  /*   LINSUM_DATA_SETUP_CUDA; */

  /*   startTimer(); */
  /*   for (RepIndex_type irep = 0; irep < run_reps; ++irep) { */

  /*     const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size); */
  /*     linsum<<<grid_size, block_size>>>( y, x, a, */
  /*                                       iend ); */ 

  /*   } */
  /*   stopTimer(); */

  /*   LINSUM_DATA_TEARDOWN_CUDA; */

  /* } else if ( vid == RAJA_CUDA ) { */

  /*   LINSUM_DATA_SETUP_CUDA; */

  /*   startTimer(); */
  /*   for (RepIndex_type irep = 0; irep < run_reps; ++irep) { */

  ///*     RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( */
  /*       RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) { */
  /*       LINSUM_BODY; */
  /*     }); */

  /*   } */
  /*   stopTimer(); */

  /*   LINSUM_DATA_TEARDOWN_CUDA; */

  /* } else { */
  /*    std::cout << "\n  LINSUM : Unknown Cuda variant id = " << vid << std::endl; */
  /* } */
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
