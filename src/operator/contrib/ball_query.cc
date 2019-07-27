/*!
 * Copyright (c) 2017 by Contributors
 * \file ball_query.cc
 * \brief ball query for 3d points
 * \author Jianlin Liu
*/
#include "./ball_query-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(BallQueryParam);

NNVM_REGISTER_OP(_contrib_BallQuery)
.describe("BallQuery foward.")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<BallQueryParam>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"xyz", "query"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"idx"};
})
.set_attr<nnvm::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  const BallQueryParam param = nnvm::get<BallQueryParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2) << "Input:[xyz, query]";

  mxnet::TShape dshape = in_shape->at(0);
  CHECK_EQ(dshape.ndim(), 3) << "xyz should be (b, n, 3)";
  CHECK_EQ(dshape[2], 3) << "xyz should be (b, n, 3)";

  mxnet::TShape qshape = in_shape->at(1);
  CHECK_EQ(qshape.ndim(), 3) << "query should be of shape (b, m, 3)";
  CHECK_EQ(qshape[2], 3) << "query should be of shape (b, m, 3)";
  // out: [b,m,nsample]
  out_shape->clear();
  out_shape->push_back(Shape3(qshape[0], qshape[1], param.nsample));
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2);
  int dtype = (*in_type)[0];
  CHECK_EQ(dtype, (*in_type)[1]);
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(mshadow::kInt32);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", BallQueryForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("xyz", "NDArray-or-Symbol", "Points xyz, 3D tensor")
.add_argument("query", "NDArray-or-Symbol", "Query point xyz, 3D tensor")
.add_arguments(BallQueryParam::__FIELDS__());

}
}