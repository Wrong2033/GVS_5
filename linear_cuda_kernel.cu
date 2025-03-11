#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void linear_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        output[row * out_features + col] = sum + bias[col];
    }
}

torch::Tensor linear_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {

    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, torch::device(torch::kCUDA));

    dim3 threads(16, 16);
    dim3 blocks(
        (batch_size + threads.x - 1) / threads.x,
        (out_features + threads.y - 1) / threads.y
    );

    linear_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    // Проверка ошибок CUDA
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaDeviceSynchronize();

    return output;
}
