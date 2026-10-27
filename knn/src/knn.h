#pragma once
#include "cpu/vision.h"
 
#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <torch/extension.h>
#endif
 
int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{
    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);
 
    float *ref_dev = ref.data_ptr<float>();
    float *query_dev = query.data_ptr<float>();
    long *idx_dev = idx.data_ptr<long>();
 
    if (ref.is_cuda()) {
#ifdef WITH_CUDA
        // TODO raise error if not compiled with CUDA
        auto dist_dev = at::empty({ref_nb * query_nb}, ref.options().device(at::kCUDA));
        float *dist_dev_ptr = dist_dev.data_ptr<float>();
 
        for (int b = 0; b < batch; b++) {
            knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
                       dist_dev_ptr, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
        }
 
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error in knn: %s\n", cudaGetErrorString(err));
            AT_ERROR("aborting");
        }
        return 1;
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
 
    auto dist_dev = at::empty({ref_nb * query_nb}, ref.options().device(at::kCPU));
    float *dist_dev_ptr = dist_dev.data_ptr<float>();
    auto ind_buf = at::empty({ref_nb}, ref.options().dtype(at::kLong).device(at::kCPU));
    long *ind_buf_ptr = ind_buf.data_ptr<long>();
 
    for (int b = 0; b < batch; b++) {
        knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
                dist_dev_ptr, idx_dev + b * k * query_nb, ind_buf_ptr);
    }
 
    return 1;
}