/*!
 * Copyright (c) 2017 by Contributors
 * \file farthest_point_sampling.cc
 * \brief farthest point sampling
 * \author Feng Wang
*/
#include "./farthest_point_sampling-inl.h"

namespace mxnet {
namespace op {

template <>
void FarthestPointSamplingForward<cpu>(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs) {
  LOG(FATAL) << "NotImplemented";
}

DMLC_REGISTER_PARAMETER(FarthestPointSamplingParam);

NNVM_REGISTER_OP(_contrib_FarthestPointSampling)
.describe("FarthestPointSampling foward.")
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<FarthestPointSamplingParam>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"idx", "min_dis"};
})
.set_attr<nnvm::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  const FarthestPointSamplingParam param = nnvm::get<FarthestPointSamplingParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1) << "Input:[data]";

  mxnet::TShape dshape = in_shape->at(0);
  CHECK_EQ(dshape.ndim(), 3) << "data should be (b, n, c), with c >= 3";
  CHECK_GE(dshape[2], 3) << "data should be (b, n, c), with c >= 3";

  // out: [b,m], [b,n]
  out_shape->clear();
  out_shape->push_back(Shape2(dshape[0], param.npoints));
  out_shape->push_back(Shape2(dshape[0], dshape[1]));
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 1);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(mshadow::kInt32);
  out_type->push_back(dtype);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", FarthestPointSamplingForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "Points data, 3D tensor whose first 3 channels are xyz")
.add_arguments(FarthestPointSamplingParam::__FIELDS__());

}
}
