/*!
 * Copyright (c) 2017 by Contributors
 * \file farthest_point_sampling.cu
 * \brief farthest point sampling
 * \author Feng Wang
*/
#include "./farthest_point_sampling-inl.h"
#include "../../common/cuda_utils.h"

namespace mxnet {
namespace op {

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

template <typename DType>
__device__ void __update(DType* __restrict__ dists, int* __restrict__ dists_i,
                            int idx1, int idx2) {
    const DType v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <typename DType, unsigned int block_size>
__global__ void farthest_point_sampling_kernel(
    int b, int n, int m, int c, const DType* __restrict__ bottom_data,
    DType* __restrict__ temp, int* __restrict__ idxs) {
    // bottom_data: (B, N, C)
    // temp: (B, N)
    // output:
    // idxs: (B, M)

    if (m <= 0) return;
    __shared__ DType dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    bottom_data += batch_index * n * c;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0) idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
    int besti = 0;
    DType best = -1;
    DType x1 = bottom_data[old * c + 0];
    DType y1 = bottom_data[old * c + 1];
    DType z1 = bottom_data[old * c + 2];
    for (int k = tid; k < n; k += stride) {
        DType x2, y2, z2;
        x2 = bottom_data[k * c + 0];
        y2 = bottom_data[k * c + 1];
        z2 = bottom_data[k * c + 2];
        // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
        // if (mag <= 1e-3)
        // continue;

        DType d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        DType d2 = min(d, temp[k]);
        temp[k] = d2;
        besti = d2 > best ? k : besti;
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
        __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
    }
}

template <>
void FarthestPointSamplingForward<gpu>(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<TBlob>& in_data,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<TBlob>& out_data) {
    using namespace mshadow;
    size_t expected_in = 1;
    size_t expected_out = 2;
    CHECK_EQ(in_data.size(), expected_in);
    CHECK_EQ(out_data.size(), expected_out);
    CHECK_EQ(out_data[farthest_point_sampling::kOut].shape_[0],
            in_data[farthest_point_sampling::kData].shape_[0]);
    CHECK_EQ(out_data[farthest_point_sampling::kMinDist].shape_[0],
            in_data[farthest_point_sampling::kData].shape_[0]);

    const FarthestPointSamplingParam param =
        nnvm::get<FarthestPointSamplingParam>(attrs.parsed);

    const int b = in_data[farthest_point_sampling::kData].size(0);
    const int n = in_data[farthest_point_sampling::kData].size(1);
    const int c = in_data[farthest_point_sampling::kData].size(2);
    const int m = param.npoints;

    Stream<gpu>* s = ctx.get_stream<gpu>();
    auto stream = mshadow::Stream<gpu>::GetStream(s);
    // assume all the data and gradient have the same type
    MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    unsigned int n_threads = opt_n_threads(n);
    const DType* input_data = in_data[farthest_point_sampling::kData].dptr<DType>();
    int* idxs = out_data[farthest_point_sampling::kOut].dptr<int>();
    DType* temp = out_data[farthest_point_sampling::kMinDist].dptr<DType>();

    Fill<false>(s, out_data[farthest_point_sampling::kMinDist], kWriteTo, std::numeric_limits<DType>::max());

    switch (n_threads) {
        case 1024:
        farthest_point_sampling_kernel<DType, 1024><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 512:
        farthest_point_sampling_kernel<DType, 512><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 256:
        farthest_point_sampling_kernel<DType, 256><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 128:
        farthest_point_sampling_kernel<DType, 128><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 64:
        farthest_point_sampling_kernel<DType, 64><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 32:
        farthest_point_sampling_kernel<DType, 32><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 16:
        farthest_point_sampling_kernel<DType, 16><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 8:
        farthest_point_sampling_kernel<DType, 8><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 4:
        farthest_point_sampling_kernel<DType, 4><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 2:
        farthest_point_sampling_kernel<DType, 2><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        case 1:
        farthest_point_sampling_kernel<DType, 1><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
        break;
        default:
        farthest_point_sampling_kernel<DType, 512><<<b, n_threads, 0, stream>>>(b, n, m, c, input_data, temp, idxs);
    }
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        LOG(FATAL) << "CUDA kernel failed : " << cudaGetErrorString(err);
        exit(-1);
    }
    })
  }

NNVM_REGISTER_OP(_contrib_FarthestPointSampling)
.set_attr<FCompute>("FCompute<gpu>", FarthestPointSamplingForward<gpu>);

}
}