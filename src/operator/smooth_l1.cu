/*!
 * Copyright (c) 2015 by Contributors
 * \file smooth_l1.cu
 * \brief Smooth L1 loss
 * \author Jian Guo
*/

#include "./smooth_l1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SmoothL1Param param) {
  return new SmoothL1Op<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
