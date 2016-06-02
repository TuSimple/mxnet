/*!
 * Copyright (c) 2016 by Contributors
 * \file max_pooling_mask-inl.h
 * \brief
 * \author Pengfei Chen
*/

#ifndef MXNET_OPERATOR_MAX_POOLING_MASK_INL_H_
#define MXNET_OPERATOR_MAX_POOLING_MASK_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace max_pool_mask_enum {
enum MaxPoolingMaskOpInputs {kData};
enum MaxPoolingMaskOpOutputs {kOut, kMask};
}  // namespace max_pool_mask_enum

struct MaxPoolingMaskParam : public dmlc::Parameter<MaxPoolingMaskParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pool_type;
  DMLC_DECLARE_PARAMETER(MaxPoolingMaskParam) {
    DMLC_DECLARE_FIELD(kernel)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("pooling kernel size: (y, x)");

    int stride_shape[] = {1, 1};
    DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 2))
    .set_expect_ndim(2).enforce_nonzero()
    .describe("stride: for pooling (y, x)");

    int pad_shape[] = {0, 0};
    DMLC_DECLARE_FIELD(pad).set_default(TShape(pad_shape, pad_shape + 2))
    .set_expect_ndim(2)
    .describe("pad for pooling: (y, x)");
  }
};

template<typename xpu, typename Reducer>
class MaxPoolingMaskOp : public Operator {
 public:
  explicit MaxPoolingMaskOp(MaxPoolingMaskParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[max_pool_mask_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[max_pool_mask_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> mask = out_data[max_pool_mask_enum::kMask].get<xpu, 4, real_t>(s);
    mshadow::Shape<2> out_shape = Shape2(out.shape_[2], out.shape_[3]);
    CHECK_EQ(param_.stride[0], param_.stride[1])
        << "Only same stride is supported now";
    Assign(out,
           req[max_pool_mask_enum::kOut],
           pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                         out_shape,
                         param_.kernel[0],
                         param_.kernel[1],
                         param_.stride[0],
                         param_.stride[1]));
    Assign(mask,
           req[max_pool_mask_enum::kMask],
           max_pool_mask<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                         out_shape,
                         param_.kernel[0],
                         param_.kernel[1],
                         param_.stride[0]));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 2);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[max_pool_mask_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data = in_data[max_pool_mask_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> output_data = out_data[max_pool_mask_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> input_grad = in_grad[max_pool_mask_enum::kData].get<xpu, 4, real_t>(s);

    mshadow::Shape<2> in_shape = Shape2(data.shape_[2], data.shape_[3]);

    Assign(input_grad, req[max_pool_mask_enum::kData],
           crop(unpool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                                pad(output_data, 0, 0),
                                pad(grad, 0, 0),
                                param_.kernel[0],
                                param_.kernel[1],
                                param_.stride[0],
                                param_.stride[1]),
                in_shape,
                param_.pad[0],
                param_.pad[1]));
  }

 private:
  MaxPoolingMaskParam param_;
};  // class MaxPoolingMaskOp

template<typename xpu>
Operator* CreateOp(MaxPoolingMaskParam param);


#if DMLC_USE_CXX11
class MaxPoolingMaskProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4) << \
                               "MaxPoolingMask: Input data should be 4D in (batch, channel, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    oshape[2] = std::min(dshape[2] + 2 * param_.pad[0] - param_.kernel[0] + param_.stride[0] - 1,
                         dshape[2] + 2 * param_.pad[0] - 1) / param_.stride[0] + 1;
    oshape[3] = std::min(dshape[3] + 2 * param_.pad[1] - param_.kernel[1] + param_.stride[1] - 1,
                         dshape[3] + 2 * param_.pad[1] - 1) / param_.stride[1] + 1;
    CHECK(oshape[2] > 0 && oshape[3] > 0) << "MaxPoolingMask: kernel size exceed input";
    out_shape->clear();
    for (int i = 0; i < 2; ++i) {
      out_shape->push_back(oshape);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    MaxPoolingMaskProp *prop_sym = new MaxPoolingMaskProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "MaxPoolingMask";
  }

  int NumOutputs() const override {
      return 2;
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[max_pool_mask_enum::kOut], in_data[max_pool_mask_enum::kData], out_data[max_pool_mask_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{in_data[max_pool_mask_enum::kData], in_grad[max_pool_mask_enum::kData]}};
#endif
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  MaxPoolingMaskParam param_;
};  // class MaxPoolingMaskProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MAX_POOLING_MASK_INL_H_

