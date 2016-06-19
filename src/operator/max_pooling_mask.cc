/*!
 * Copyright (c) 2016 by Contributors
 * \file max_pooling_mask.cc
 * \brief
 * \author Pengfei Chen
*/
#include "./max_pooling_mask-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MaxPoolingMaskParam param) {
  return new MaxPoolingMaskOp<cpu, mshadow::red::maximum>(param);
}

Operator* MaxPoolingMaskProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MaxPoolingMaskParam);

MXNET_REGISTER_OP_PROPERTY(MaxPoolingMask, MaxPoolingMaskProp)
.describe("Perform spatial pooling on inputs.")
.add_argument("data", "Symbol", "Input data to the pooling operator.")
.add_arguments(MaxPoolingMaskParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

