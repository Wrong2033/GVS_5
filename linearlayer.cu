#include <torch/extension.h>

template<typename scalar_t>
using accessor_2d = torch::PackedTensorAccessor32<scalar_t,2>;

template<typename scalar_t>
using accessor_1d = torch::PackedTensorAccessor32<scalar_t,1>;

template<typename scalar_t>
__global__ void linear_fwd_kern (const accessor_2d<scalar_t> input,
                                const accessor_2d<scalar_t> weight,
                                const accessor_1d<scalar_t> bias,
                                accessor_2d<scalar_t> output)
{
    auto n = blockDim.x * blockIdx.x + threadIdx.x;
    auto m = blockDim.y * blockIdx.y + threadIdx.y;

    scalar_t acc = 0;
    if (m < input.size(0) && n < weight.size(0)) {
        for (int k = 0; k < input.size(1); k++) {
            acc += input[m][k] * weight[n][k];
        }

        output[m][n] = acc + bias[n];
    }
}

template<typename scalar_t>
__global__ void linear_bwd_kern (const accessor_2d<scalar_t> input,
                                const accessor_2d<scalar_t> weight,
                                const accessor_2d<scalar_t> d_output,
                                accessor_2d<scalar_t> d_input,
                                accessor_2d<scalar_t> d_weight,
                                accessor_1d<scalar_t> d_bias)
{
    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    auto m = blockIdx.y * blockDim.y + threadIdx.y;

    // n - weight.size(1), m - d_output.size(0)

    /* dX = dY @ W */
    if (m < d_output.size(0) && n < weight.size(1)) {
        scalar_t acc = 0;
        for (int k = 0; k < d_output.size(1); k++) {
            acc += d_output[m][k] * weight[k][n];
        }
        d_input[m][n] = acc;
    }

    /* dW = dY^T @ X */
    if (m < input.size(1) && n < d_output.size(1)) {
        scalar_t acc = 0;
        for (int k = 0; k < d_output.size(0); k++) {
            acc += d_output[k][n] * input[k][m];
        }
        d_weight[n][m] = acc;
    }

    /* db = SUM(dY, m) */
    if (n < d_output.size(1) && m < d_output.size(0)) {
        atomicAdd(&d_bias[n], d_output[m][n]);
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int kBlockSize = 32;

__forceinline__ int calc_grid_size(int block_size, int m) {
    return (m + block_size - 1) / block_size;
}

torch::Tensor forward_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // проверки на введенные переменные
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    // ввод x(m,k), w(n,k), b(n)
    int n = bias.numel();
    int k = weight.numel() / n;
    int m = input.numel() / k;

    // вывод y(m,n)
    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCUDA).requires_grad(true);
    torch::Tensor output = torch::zeros({m, n}, options);

    constexpr dim3 dimBlock = {kBlockSize, kBlockSize};
    const dim3 dimGrid = {calc_grid_size(kBlockSize, n), calc_grid_size(kBlockSize, m)};

    linear_fwd_kern<<<dimGrid, dimBlock>>>(
        input.packed_accessor32<float, 2>(),
        weight.packed_accessor32<float, 2>(),
        bias.packed_accessor32<float, 1>(),
        output.packed_accessor32<float, 2>()
    );

    return output;
}

std::vector<torch::Tensor> backward_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor d_output) {
    // Проверка входных тензеров
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(d_output);

    // Инициализируем переменные
    auto batch_size = input.size(0);
    auto weight_rows = weight.size(0);
    auto weight_cols = weight.size(1);

    torch::Tensor d_input = torch::zeros_like(input);
    torch::Tensor d_weight = torch::zeros_like(weight);
    torch::Tensor d_bias = torch::zeros_like(bias);

    constexpr dim3 dimBlock = {kBlockSize, kBlockSize};
    const dim3 dimGrid = {
        calc_grid_size(dimBlock.x, weight_cols),
        calc_grid_size(dimBlock.y, std::max({weight_cols, batch_size}))
    };

    linear_bwd_kern<<<dimGrid, dimBlock>>>(
        input.packed_accessor32<float, 2>(),
        weight.packed_accessor32<float, 2>(),
        d_output.packed_accessor32<float, 2>(),
        d_input.packed_accessor32<float, 2>(),
        d_weight.packed_accessor32<float, 2>(),
        d_bias.packed_accessor32<float, 1>()
    );

    return std::vector<torch::Tensor>{d_input, d_weight, d_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_forward_linear", &forward_linear, "Custom function forward linear layer");
    m.def("my_backward_linear", &backward_linear, "Custom function backward linear layer");
}