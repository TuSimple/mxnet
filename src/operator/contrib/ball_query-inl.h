/*!
 * Copyright (c) 2017 by Contributors
 * \file ball_query-inl.h
 * \brief ball query for 3d points
 * \author Jianlin Liu
*/
#ifndef MXNET_OPERATOR_CONTRIB_BALL_QUERY_INL_H_
#define MXNET_OPERATOR_CONTRIB_BALL_QUERY_INL_H_

#include <vector>
#include <utility>
#include <mxnet/operator_util.h>
#include "../mxnet_op.h"
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../tensor/init_op.h"
#include "../operator_common.h"

namespace mxnet {
  typedef std::vector<mxnet::TShape> ShapeVector;
namespace op {

struct BallQueryParam : public dmlc::Parameter<BallQueryParam> {
  float radius;
  int nsample;
  DMLC_DECLARE_PARAMETER(BallQueryParam) {
    DMLC_DECLARE_FIELD(radius)
      .describe("Search radius.");
    DMLC_DECLARE_FIELD(nsample)
      .describe("Number of samples ball within radius to be returned.");
  }
};

struct BallQueryKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const int n, const int m,
                                  const DType* xyz, const DType* query, int* idx,
                                  const float r, const int nsample) {
     int b = i / m;
     query += i * 3;
     xyz += b * n * 3;
     idx += i * nsample;

     float r2 = r * r;
     float q_x = query[0];
     float q_y = query[1];
     float q_z = query[2];

     for(int cnt=0, k=0; k < n; k++) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (q_x - x) * (q_x - x) + (q_y - y) * (q_y - y) + (q_z - z) * (q_z - z);
        if (d2 < r2) {
           if (cnt == 0){
              for (int l = 0; l < nsample; ++l) {
                idx[l] = k;
              }
           }
           idx[cnt++] = k;
           if (cnt >= nsample) break;
        }
     }
  }
};

template <typename xpu>
void BallQueryForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                         const std::vector<TBlob>& in_data,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  CHECK_EQ(in_data.size(), 2U);
  CHECK_EQ(out_data.size(), 1U);

  const int batch_size = in_data[0].size(0);
  const int n = in_data[0].size(1);
  const int m = in_data[1].size(1);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const BallQueryParam& param = nnvm::get<BallQueryParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(in_data[0].type_flag_, DType, {
     mxnet_op::Kernel<BallQueryKernel, xpu>::Launch(
       s, batch_size*m, n, m, in_data[0].dptr<DType>(), in_data[1].dptr<DType>(), out_data[0].dptr<int>(),
       param.radius, param.nsample);
  });
}

}
}

#endif  // MXNET_OPERATOR_CONTRIB_BALL_QUERY_INL_H_