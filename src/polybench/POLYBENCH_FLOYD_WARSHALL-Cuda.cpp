  
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_CUDA \
  Real_ptr pin; \
  Real_ptr pout; \
\
  allocAndInitCudaDeviceData(pin, m_pin, m_N * m_N); \
  allocAndInitCudaDeviceData(pout, m_pout, m_N * m_N);


#define POLYBENCH_FLOYD_WARSHALL_TEARDOWN_CUDA \
  getCudaDeviceData(m_pout, pout, m_N * m_N); \
  deallocCudaDeviceData(pin); \
  deallocCudaDeviceData(pout);


__global__ void poly_floyd_warshall(Real_ptr pout, Real_ptr pin,
                                    Index_type k,
                                    Index_type N)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   POLYBENCH_FLOYD_WARSHALL_BODY;              
}


void POLYBENCH_FLOYD_WARSHALL::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;


  if ( vid == Base_CUDA ) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type k = 0; k < N; ++k) {

        dim3 nblocks1(N, 1, 1);
        dim3 nthreads_per_block1(1, N, 1);
        poly_floyd_warshall<<<nblocks1, nthreads_per_block1>>>(pout, pin,
                                                               k, N);

      }

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_CUDA;

    POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::CudaKernelAsync<
            RAJA::statement::For<1, RAJA::cuda_block_exec,
              RAJA::statement::For<2, RAJA::cuda_thread_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N}),
        [=] __device__ (Index_type k, Index_type i, Index_type j) {
          POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  