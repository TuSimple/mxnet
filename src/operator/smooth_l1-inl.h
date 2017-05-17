/*!
 * Copyright (c) 2015 by Contributors
 * \file smooth_l1-inl.h
 * \brief Smooth L1 loss
 * \author Jian Guo
*/

#ifndef MXNET_OPERATOR_SMOOTH_L1_INL_H_
#define MXNET_OPERATOR_SMOOTH_L1_INL_H_

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

namespace mshadow_op {
/* Smooth L1 Loss is a loss specific for R-CNN franchise training
 * Smooth L1 Loss function
 * f(x) = 0.5 * (sigma * x) ^ 2,     x < 1 / sigma^2
 *      = |x| - 0.5 / sigma / sigma, otherwise
 * When sigma = 1, it is equivalent to Huber Loss evaluated at
 * delta = 1.
 * smooth_l1_loss = w_out * f(w_in * x)
 * with w_in, w_out provided by input_data.
 */
struct smooth_l1_loss {
  // a is x, b is sigma2
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    if (a > 1.0f / b) {
      return a - 0.5f / b;
    } else if (a < -1.0f / b) {
      return -a - 0.5f / b;
    } else {
      return 0.5f * a * a * b;
    }
  }
};  // struct smooth_l1_loss

/* The derivative of smooth l1 loss is
 * f'(x) = sigma^2 * x, x < 1 / sigma^2
 *       = sign(x),     otherwise
 */
struct smooth_l1_derivative {
  // a is x, b is sigma2
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    if (a > 1.0f / b) {
      return 1.0f;
    } else if (a < -1.0f / b) {
      return -1.0f;
    } else {
      return b * a;
    }
  }
};  // struct smooth_l1_derivative
}  // namespace mshadow_op

namespace smooth_l1_enum {
enum SmoothL1OpInputs {kData, kTarget, kInsideWeight, kOutsideWeight};
enum SmoothL1OpOutputs {kOut};
}

struct SmoothL1Param : public dmlc::Parameter<SmoothL1Param> {
  float sigma;
  float grad_scale;
  size_t num_args;
  DMLC_DECLARE_PARAMETER(SmoothL1Param) {
    DMLC_DECLARE_FIELD(sigma)
    .set_default(1.0f)
    .describe("The reciprocal of square sigma is the turning point of smooth l1 loss.");
    DMLC_DECLARE_FIELD(grad_scale)
    .set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(num_args)
    .set_default(2)
    .describe("Arguments are [data, target]. optional: [inside_weight, outside_weight]");
  }
};

template<typename xpu>
Operator* CreateOp(SmoothL1Param param);

template<typename xpu>
class SmoothL1Op : public Operator {
 public:
  explicit SmoothL1Op(SmoothL1Param param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), param_.num_args);
    if (in_data.size() >= 3) {
      // inside_weight and outside_weight are available
      Stream<xpu> *s = ctx.get_stream<xpu>();

      Tensor<xpu, 2> data = in_data[smooth_l1_enum::kData].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> target = in_data[smooth_l1_enum::kTarget].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> inside_weight =
                     in_data[smooth_l1_enum::kInsideWeight].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> outside_weight =
                     in_data[smooth_l1_enum::kOutsideWeight].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> out = out_data[smooth_l1_enum::kOut].get<xpu, 2, real_t>(s);

      real_t sigma2 = param_.sigma * param_.sigma;
      out = outside_weight *
            F<mshadow_op::smooth_l1_loss>((data - target) * inside_weight, sigma2);
    } else {
      // by default all the weights would be 1.0
      Stream<xpu> *s = ctx.get_stream<xpu>();

      Tensor<xpu, 2> data = in_data[smooth_l1_enum::kData].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> target = in_data[smooth_l1_enum::kTarget].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> out = out_data[smooth_l1_enum::kOut].get<xpu, 2, real_t>(s);

      real_t sigma2 = param_.sigma * param_.sigma;
      out = F<mshadow_op::smooth_l1_loss>((data - target), sigma2);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), param_.num_args);
    if (in_data.size() >= 3) {
      // inside_weight and outside_weight are available
      Stream<xpu> *s = ctx.get_stream<xpu>();

      Tensor<xpu, 2> data = in_data[smooth_l1_enum::kData].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> target = in_data[smooth_l1_enum::kTarget].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> inside_weight =
                     in_data[smooth_l1_enum::kInsideWeight].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> outside_weight =
                     in_data[smooth_l1_enum::kOutsideWeight].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> grad = in_grad[smooth_l1_enum::kData].get<xpu, 2, real_t>(s);

      real_t sigma2 = param_.sigma * param_.sigma;
      grad = param_.grad_scale * outside_weight * inside_weight *
             F<mshadow_op::smooth_l1_derivative>((data - target), sigma2);
    } else {
      // by default all the weights would be 1.0
      Stream<xpu> *s = ctx.get_stream<xpu>();

      Tensor<xpu, 2> data = in_data[smooth_l1_enum::kData].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> target = in_data[smooth_l1_enum::kTarget].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> grad = in_grad[smooth_l1_enum::kData].get<xpu, 2, real_t>(s);

      real_t sigma2 = param_.sigma * param_.sigma;
      grad = param_.grad_scale * F<mshadow_op::smooth_l1_derivative>((data - target), sigma2);
    }
  }

 private:
  SmoothL1Param param_;
};  // class SmoothL1Op

#if DMLC_USE_CXX11
class SmoothL1Prop : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (param_.num_args == 2) {
      return {"data", "target"};
    } else {
      return {"data", "target", "inside_weight", "outside_weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (in_shape->size() == 4) {
      CHECK_EQ(in_shape->size(), param_.num_args) <<
        "Input: [data, target, inside_weight, outside_weight]";
    } else if (in_shape->size() == 2) {
      CHECK_EQ(in_shape->size(), param_.num_args) << "Input: [data, target]";
    } else {
      CHECK_EQ(in_shape->size(), param_.num_args) << "Input size should equal num_args";
    }
    const TShape &shape = in_shape->at(smooth_l1_enum::kData);

    out_shape->clear();
    out_shape->push_back(shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto smooth_l1_sym = new SmoothL1Prop();
    smooth_l1_sym->param_ = param_;
    return smooth_l1_sym;
  }

  std::string TypeString() const override {
    return "SmoothL1";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.num_args == 2) {
      return {in_data[smooth_l1_enum::kData], in_data[smooth_l1_enum::kTarget]};
    } else {
      return {in_data[smooth_l1_enum::kData], in_data[smooth_l1_enum::kTarget],
              in_data[smooth_l1_enum::kInsideWeight], in_data[smooth_l1_enum::kOutsideWeight]};
    }
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  SmoothL1Param param_;
};  // class SmoothL1Prop
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SMOOTH_L1_INL_H_
