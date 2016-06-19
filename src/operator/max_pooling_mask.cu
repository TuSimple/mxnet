/*!
 * Copyright (c) 2016 by Contributors
 * \file max_pooling_mask.cu
 * \brief
 * \author Pengfei Chen
*/

// #include "./pooling_mask-inl.h"
// #if MXNET_USE_CUDNN == 1
// #include "./cudnn_pooling-inl.h"
// #endif  // MXNET_USE_CUDNN
#include "./max_pooling_mask-inl.h"
 
namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MaxPoolingMaskParam param) {
// #if MXNET_USE_CUDNN == 1
//   return new CuDNNPoolingMaskOp(param);
// #else
    return new MaxPoolingMaskOp<gpu, mshadow::red::maximum>(param);
// #endif  // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet

