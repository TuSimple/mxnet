/*!
 * Copyright (c) 2017 by Contributors
 * \file point_pooling-inl.h
 * \brief farthest point sampling
 * \author Feng Wang
*/
#ifndef MXNET_OPERATOR_CONTRIB_FARTHEST_POINT_SAMPLING_INL_H_
#define MXNET_OPERATOR_CONTRIB_FARTHEST_POINT_SAMPLING_INL_H_

#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../tensor/init_op.h"
#include "../operator_common.h"

namespace mxnet {
typedef std::vector<mxnet::TShape> ShapeVector;
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace farthest_point_sampling {
enum FarthestPointSamplingOpInputs { kData };
enum FarthestPointSamplingOpOutputs { kOut, kMinDist };
}  // namespace farthest_point_sampling

struct FarthestPointSamplingParam
    : public dmlc::Parameter<FarthestPointSamplingParam> {
  int npoints;
  DMLC_DECLARE_PARAMETER(FarthestPointSamplingParam) {
    DMLC_DECLARE_FIELD(npoints).describe("Number of keypoints.");
  }
};

template <typename xpu>
void FarthestPointSamplingForward(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& in_data,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& out_data);
}
}

#endif  // MXNET_OPERATOR_CONTRIB_FARTHEST_POINT_SAMPLING_INL_H_
