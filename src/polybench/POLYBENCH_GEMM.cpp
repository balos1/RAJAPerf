//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{


POLYBENCH_GEMM::POLYBENCH_GEMM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GEMM, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_ni = 20; m_nj = 25; m_nk = 30;
      run_reps = 10000;
      break;
    case Small:
      m_ni = 60; m_nj = 70; m_nk = 80;
      run_reps = 1000;
      break;
    case Medium:
      m_ni = 200; m_nj = 220; m_nk = 240;
      run_reps = 100;
      break;
    case Large:
      m_ni = 1000; m_nj = 1100; m_nk = 1200;
      run_reps = 1;
      break;
    case Extralarge:
      m_ni = 2000; m_nj = 2300; m_nk = 2600;
      run_reps = 1;
      break;
    default:
      m_ni = 200; m_nj = 220; m_nk = 240;
      run_reps = 100;
      break;
  }

  setDefaultSize( m_ni * (m_nj + m_nj*m_nk) );
  setDefaultReps(run_reps);

  m_alpha = 0.62;
  m_beta = 1.002;
}

POLYBENCH_GEMM::~POLYBENCH_GEMM() 
{

}

void POLYBENCH_GEMM::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitDataConst(m_C, m_ni * m_nj, 0.0, vid);
}

void POLYBENCH_GEMM::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_GEMM_DATA_SETUP;

  auto poly_gemm_base_lam2 = [=](Index_type i, Index_type j) {
                               POLYBENCH_GEMM_BODY2;
                             };
  auto poly_gemm_base_lam3 = [=](Index_type i, Index_type j, Index_type k,
                                 Real_type& dot) {
                               POLYBENCH_GEMM_BODY3;
                              };
  auto poly_gemm_base_lam4 = [=](Index_type i, Index_type j,
                                 Real_type& dot) {
                               POLYBENCH_GEMM_BODY4;
                              };

  POLYBENCH_GEMM_VIEWS_RAJA;

  auto poly_gemm_lam1 = [=](Index_type /*i*/, Index_type /*j*/, Index_type /*k*/, 
                            Real_type& dot) {
                            POLYBENCH_GEMM_BODY1_RAJA;
                           };
  auto poly_gemm_lam2 = [=](Index_type i, Index_type j, Index_type /*k*/, 
                            Real_type& /*dot*/) {
                            POLYBENCH_GEMM_BODY2_RAJA;
                           };
  auto poly_gemm_lam3 = [=](Index_type i, Index_type j, Index_type k, 
                            Real_type& dot) {
                            POLYBENCH_GEMM_BODY3_RAJA;
                           };
  auto poly_gemm_lam4 = [=](Index_type i, Index_type j, Index_type /*k*/, 
                            Real_type& dot) {
                            POLYBENCH_GEMM_BODY4_RAJA;
                           };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < ni; ++i ) { 
          for (Index_type j = 0; j < nj; ++j ) {
            POLYBENCH_GEMM_BODY1;
            POLYBENCH_GEMM_BODY2;
            for (Index_type k = 0; k < nk; ++k ) {
               POLYBENCH_GEMM_BODY3;
            }
            POLYBENCH_GEMM_BODY4;
          }
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < ni; ++i ) {
          for (Index_type j = 0; j < nj; ++j ) {
            POLYBENCH_GEMM_BODY1;
            poly_gemm_base_lam2(i, j);
            for (Index_type k = 0; k < nk; ++k ) {
              poly_gemm_base_lam3(i, j, k, dot);
            }
            poly_gemm_base_lam4(i, j, dot);
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>,
              RAJA::statement::Lambda<1>,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<2>
              >,
              RAJA::statement::Lambda<3>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
     
          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),  // variable for dot

          poly_gemm_lam1,
          poly_gemm_lam2,
          poly_gemm_lam3,
          poly_gemm_lam4

        );

      }
      stopTimer();

      break;
    }

#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for collapse(2)
        for (Index_type i = 0; i < ni; ++i ) {
          for (Index_type j = 0; j < nj; ++j ) {
            POLYBENCH_GEMM_BODY1;
            POLYBENCH_GEMM_BODY2;
            for (Index_type k = 0; k < nk; ++k ) {
              POLYBENCH_GEMM_BODY3;
            }
            POLYBENCH_GEMM_BODY4;
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for collapse(2)
        for (Index_type i = 0; i < ni; ++i ) { 
          for (Index_type j = 0; j < nj; ++j ) {
            POLYBENCH_GEMM_BODY1;
            poly_gemm_base_lam2(i, j);
            for (Index_type k = 0; k < nk; ++k ) {
              poly_gemm_base_lam3(i, j, k, dot);
            }
            poly_gemm_base_lam4(i, j, dot);
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::Lambda<0>,
            RAJA::statement::Lambda<1>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<2>
            >,
            RAJA::statement::Lambda<3>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
     
          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),  // variable for dot

          poly_gemm_lam1,
          poly_gemm_lam2,
          poly_gemm_lam3,
          poly_gemm_lam4

        );

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  POLYBENCH_GEMM : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_GEMM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_C, m_ni * m_nj);
}

void POLYBENCH_GEMM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
}

} // end namespace polybench
} // end namespace rajaperf
