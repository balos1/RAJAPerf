//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// LINSUM kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   Z[i] = a*X[i] + b*Y[i];
/// }
///

#ifndef RAJAPerf_Basic_LINSUM_HPP
#define RAJAPerf_Basic_LINSUM_HPP

#define LINSUM_DATA_SETUP \
  Real_ptr X = m_x; \
  Real_ptr Y = m_y; \
  Real_ptr Z = m_z; \
  Real_type a = m_a; \
  Real_type b = m_a;

#define LINSUM_BODY  \
  Z[i] = a*X[i] + b*Y[i];


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class LINSUM : public KernelBase
{
public:

  LINSUM(const RunParams& params);

  ~LINSUM();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
  Real_type m_a;
  Real_type m_b;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
