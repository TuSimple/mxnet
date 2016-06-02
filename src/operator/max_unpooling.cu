/*!
 * Copyright (c) 2016 by Contributors
 * \file max_unpooling.cu
 * \brief
 * \author Pengfei Chen
*/

// #include "./deconvolution-inl.h"
#include "./max_unpooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MaxUnpoolingParam param) {
  return new MaxUnpoolingOp<gpu, mshadow::red::maximum>(param);
}

}  // namespace op
}  // namespace mxnet