//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LINSUM.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace basic
{


LINSUM::LINSUM(const RunParams& params)
  : KernelBase(rajaperf::Basic_LINSUM, params)
{
   setDefaultSize(100000);
   setDefaultReps(5000);
}

LINSUM::~LINSUM() 
{
}

void LINSUM::setUp(VariantID vid)
{
  allocAndInitDataConst(m_z, getRunSize(), 0.0, vid);
  allocAndInitData(m_x, getRunSize(), vid);
  allocAndInitData(m_y, getRunSize(), vid);
  initData(m_a);
  initData(m_b);
}

void LINSUM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_z, getRunSize());
}

void LINSUM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
}

} // end namespace basic
} // end namespace rajaperf
