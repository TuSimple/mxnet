/*!
 * Copyright (c) 2017 by Contributors
 * \file ball_query.cu
 * \brief ball query for 3d points
 * \author Jianlin Liu
*/
#include "./ball_query-inl.h"
#include "../../common/cuda_utils.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_BallQuery)
.set_attr<FCompute>("FCompute<gpu>", BallQueryForward<gpu>);

}
}