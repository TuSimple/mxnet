/*!
 * Copyright (c) 2015 by Contributors
 * \file smooth_l1.cc
 * \brief Smooth L1 loss
 * \author Jian Guo
*/

#include "./smooth_l1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SmoothL1Param param) {
  return new SmoothL1Op<cpu>(param);
}

Operator *SmoothL1Prop::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SmoothL1Param);

MXNET_REGISTER_OP_PROPERTY(SmoothL1, SmoothL1Prop)
.describe("Smooth L1 loss.")
.add_argument("data", "Symbol", "Input data for loss function")
.add_argument("target", "Symbol", "Target for loss function, of the same size as data")
.add_argument("inside_weight", "Symbol", "Scale for input data")
.add_argument("outside_weight", "Symbol", "Scale for output")
.add_arguments(SmoothL1Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet
