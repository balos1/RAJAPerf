//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf 
{
namespace polybench
{

 
POLYBENCH_GESUMMV::POLYBENCH_GESUMMV(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GESUMMV, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N = 30;
      run_reps = 10000;
      break;
    case Small:
      m_N = 90;
      run_reps = 1000;
      break;
    case Medium:
      m_N = 250;
      run_reps = 100;
      break;
    case Large:
      m_N = 1300;
      run_reps = 1;
      break;
    case Extralarge:
      m_N = 2800;
      run_reps = 1;
      break;
    default:
      m_N = 1600;
      run_reps = 120;
      break;
  }

  setDefaultSize( m_N * m_N );
  setDefaultReps(run_reps);

  m_alpha = 0.62;
  m_beta = 1.002;
}

POLYBENCH_GESUMMV::~POLYBENCH_GESUMMV() 
{

}

void POLYBENCH_GESUMMV::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_x, m_N, vid);
  allocAndInitDataConst(m_y, m_N, 0.0, vid);
  allocAndInitData(m_A, m_N * m_N, vid);
  allocAndInitData(m_B, m_N * m_N, vid);
}

void POLYBENCH_GESUMMV::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_y, m_N);
}

void POLYBENCH_GESUMMV::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_A);
  deallocData(m_B);
}

} // end namespace polybench
} // end namespace rajaperf
