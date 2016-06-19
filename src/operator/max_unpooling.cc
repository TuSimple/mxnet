/*!
 * Copyright (c) 2016 by Contributors
 * \file max_unpooling.cc
 * \brief
 * \author Pengfei Chen
*/

#include "./max_unpooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MaxUnpoolingParam param) {
  return new MaxUnpoolingOp<cpu, mshadow::red::maximum>(param);
}

Operator* MaxUnpoolingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MaxUnpoolingParam);

MXNET_REGISTER_OP_PROPERTY(MaxUnpooling, MaxUnpoolingProp)
.describe("Perform unpooling to inputs based on the pooling masks")
.add_argument("data", "Symbol[]", "Array of tensors to unpooling")
.add_arguments(MaxUnpoolingParam::__FIELDS__())
.set_key_var_num_args("num_args");
}  // namespace op
}  // namespace mxnet
