//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf 
{
namespace polybench
{

 
POLYBENCH_JACOBI_1D::POLYBENCH_JACOBI_1D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_JACOBI_1D, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N=300;
      m_tsteps=20;
      run_reps = 10000;
      break;
    case Small:
      m_N=1200;
      m_tsteps=100;
      run_reps = 1000;
      break;
    case Medium:
      m_N=4000;
      m_tsteps=100;
      run_reps = 100;
      break;
    case Large:
      m_N=200000;
      m_tsteps=50;
      run_reps = 1;
      break;
    case Extralarge:
      m_N=2000000;
      m_tsteps=10;
      run_reps = 20;
      break;
    default:
      m_N=4000000;
      m_tsteps=10;
      run_reps = 10;
      break;
  }

  setDefaultSize( m_tsteps * 2 * m_N );
  setDefaultReps(run_reps);
}

POLYBENCH_JACOBI_1D::~POLYBENCH_JACOBI_1D() 
{

}

void POLYBENCH_JACOBI_1D::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_Ainit, m_N, vid);
  allocAndInitData(m_Binit, m_N, vid);
  allocAndInitDataConst(m_A, m_N, 0.0, vid);
  allocAndInitDataConst(m_B, m_N, 0.0, vid);
}

void POLYBENCH_JACOBI_1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_A, m_N);
  checksum[vid] += calcChecksum(m_B, m_N);
}

void POLYBENCH_JACOBI_1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_Ainit);
  deallocData(m_Binit);
}

} // end namespace polybench
} // end namespace rajaperf
