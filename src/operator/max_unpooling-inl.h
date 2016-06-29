/*!
 * Copyright (c) 2016 by Contributors
 * \file max_unpooling-inl.h
 * \brief
 * \author Pengfei Chen
*/
#ifndef MXNET_OPERATOR_MAX_UNPOOLING_INL_H_
#define MXNET_OPERATOR_MAX_UNPOOLING_INL_H_

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

namespace max_unpool_enum {
enum MaxUnpoolingOpInputs {kData, kPoolMask};
enum MaxUnpoolingOpOutputs {kOut};
}  // namespace max_unpool_enum

struct MaxUnpoolingParam : public dmlc::Parameter<MaxUnpoolingParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int unpool_type;
  int num_args;
  TShape unpool_size;
  DMLC_DECLARE_PARAMETER(MaxUnpoolingParam) {
    DMLC_DECLARE_FIELD(kernel)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("unpooling kernel size: (y, x)");


    int stride_shape[] = {1, 1};
    DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 2))
    .set_expect_ndim(2).enforce_nonzero()
    .describe("stride: for pooling (y, x)");

    int pad_shape[] = {0, 0};
    DMLC_DECLARE_FIELD(pad).set_default(TShape(pad_shape, pad_shape + 2))
    .set_expect_ndim(2)
    .describe("pad for pooling: (y, x), currently only support (0,0)");

    int unpool_shape[] = {0, 0};
    DMLC_DECLARE_FIELD(unpool_size).set_default(TShape(unpool_shape, unpool_shape + 2))
    .set_expect_ndim(2)
    .describe("target size for unpooling (y, x)");

    // currently only for max unpooling
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Data used to unpooling. For max unpooling need a pooling mask.");
  }
};  // struct MaxUnpoolingParam

template<typename xpu, typename Reducer>
class MaxUnpoolingOp : public Operator {
 public:
  explicit MaxUnpoolingOp(MaxUnpoolingParam p){
    this->param_ = p;
    // unpool_size_ = this->param_.unpool_size;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "required size: 2" << "real size:" << in_data.size();
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[max_unpool_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> mask = in_data[max_unpool_enum::kPoolMask].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[max_unpool_enum::kOut].get<xpu, 4, real_t>(s);
    mshadow::Shape<2> out_shape = Shape2(out.shape_[2], out.shape_[3]);

    CHECK_EQ(param_.stride[0], param_.stride[1])
        << "Only same stride is supported now";
    CHECK_EQ(param_.pad[0], 0) 
        << "currently, only zero padding is allowed.";
    CHECK_EQ(param_.pad[1], 0) 
        << "currently, only zero padding is allowed.";
    
    Assign(out,
           req[max_unpool_enum::kOut],
           max_unpool_forward(
                         mask,
                         data,
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
    CHECK_EQ(in_grad.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(req.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[max_unpool_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> mask = in_data[max_unpool_enum::kPoolMask].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> input_grad = in_grad[max_unpool_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> mask_grad = in_grad[max_unpool_enum::kPoolMask].get<xpu, 4, real_t>(s);
    mshadow::Shape<2> mask_shape = Shape2(mask.shape_[2], mask.shape_[3]);

    Assign(input_grad, req[max_unpool_enum::kData],
              max_unpool_backward(
                              mask,
                              grad,
                              mask_shape,
                              param_.kernel[0],
                              param_.kernel[1],
                              param_.stride[0]));
    Assign(mask_grad, req[max_unpool_enum::kData],
              mask_backward<Reducer>(pad(mask, 0, 0)));
  }

 private:
  MaxUnpoolingParam param_;
  // TShape unpool_size_;
};  // class MaxUnpoolingOp

template<typename xpu>
Operator *CreateOp(MaxUnpoolingParam param);


#if DMLC_USE_CXX11
class MaxUnpoolingProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_args; ++i) {
      ret.push_back(std::string("arg") + static_cast<char>('0' + i));
    }
    return ret;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
    const TShape &dshape = (*in_shape)[0];
    const TShape unpool_size_ = param_.unpool_size;
    CHECK_EQ(dshape.ndim(), 4) << \
                               "MaxUnpooling: Input data should be 4D in (batch, channel, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    for (int i = 1; i < param_.num_args; ++i) {
        const TShape &tmp = in_shape->at(i);
        if (tmp.ndim() == 0) return false;
        for (index_t j = 0; j < dshape.ndim(); ++j) {
            CHECK_EQ(dshape[j], tmp[j])
             << "Uncompatible Shapes on dimension : "<< j << " data:"<< dshape[j] << ", mask:" <<  tmp[j];
        }
    }
    if(unpool_size_[0] <= 0 || unpool_size_[1] <= 0){
        oshape[2] = (dshape[2] - 1) * param_.stride[0] + param_.kernel[0] - 2 * param_.pad[0];
        oshape[3] = (dshape[3] - 1) * param_.stride[1] + param_.kernel[1] - 2 * param_.pad[1];
    }
    else {
        oshape[2] = unpool_size_[0];
        oshape[3] = unpool_size_[1];
    }
    CHECK(oshape[2] > 0 && oshape[3] > 0) << "Pooling: kernel size exceed input";
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MaxUnpoolingProp();
    ptr->param_ = this->param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MaxUnpooling";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
      return {out_grad[max_unpool_enum::kOut], in_data[max_unpool_enum::kData], in_data[max_unpool_enum::kPoolMask]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{in_data[max_unpool_enum::kData], in_grad[max_unpool_enum::kData]}};
#endif
  }
  Operator* CreateOperator(Context ctx) const override;

 private:
  MaxUnpoolingParam param_;
};  // class MaxUnpoolingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MAX_UNPOOLING_INL_H_
