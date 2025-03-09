#include <torch/extension.h>

template<typename scalar_t>
using accessor_2d = torch::PackedTensorAccessor32<scalar_t, 2>;

template<typename scalar_t>
using accessor_1d = torch::PackedTensorAccessor32<scalar_t, 1>;

template<typename scalar_t>
__global__ void linear_fwd_kern(const accessor_2d<scalar_t> input,
                                const accessor_2d<scalar_t> weight,
                                const accessor_1d<scalar_t> bias,
                                accessor_2d<scalar_t> output) {
    auto n = blockDim.x * blockIdx.x + threadIdx.x;
    auto m = blockDim.y * blockIdx.y + threadIdx.y;

    if (m < input.size(0) && n < weight.size(0)) {
        scalar_t acc = 0;
        for (int k = 0; k < input.size(1); k++) {
            acc += input[m][k] * weight[n][k];
        }
        output[m][n] = acc + bias[n];
    }
}

template<typename scalar_t>
__global__ void linear_bwd_kern(const accessor_2d<scalar_t> input,
                                const accessor_2d<scalar_t> weight,
                                const accessor_2d<scalar_t> d_output,
                                accessor_2d<scalar_t> d_input,
                                accessor_2d<scalar_t> d_weight,
                                accessor_1d<scalar_t> d_bias) {
    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    auto m = blockIdx.y * blockDim.y + threadIdx.y;

    // dX = dY @ W.T
    if (m < d_output.size(0) && n < weight.size(1)) {
        scalar_t acc = 0;
        for (int k = 0; k < d_output.size(1); k++) {
            acc += d_output[m][k] * weight[k][n];
        }
        d_input[m][n] = acc;
    }

    // dW = dY.T @ X
    if (m < input.size(1) && n < d_output.size(1)) {
        scalar_t acc = 0;
        for (int k = 0; k < d_output.size(0); k++) {
            acc += d_output[k][n] * input[k][m];
        }
        d_weight[n][m] = acc;
    }

    // db = sum(dY, axis=0)
    if (n < d_output.size(1)) {
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
    // Проверка входных данных
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    // Проверка размерностей
    TORCH_CHECK(input.size(1) == weight.size(1), "Input and weight dimensions do not match");
    TORCH_CHECK(weight.size(0) == bias.size(0), "Weight and bias dimensions do not match");

    // Размеры данных
    int n = weight.size(0);  // Количество выходных нейронов
    int k = weight.size(1);  // Количество входных нейронов
    int m = input.size(0);   // Размер батча

    // Инициализация выхода
    auto options = torch::TensorOptions().dtype(input.dtype()).device(torch::kCUDA).requires_grad(true);
    torch::Tensor output = torch::zeros({m, n}, options);

    // Размеры блоков и сетки
    constexpr dim3 dimBlock = {kBlockSize, kBlockSize};
    const dim3 dimGrid = {
        calc_grid_size(kBlockSize, n),
        calc_grid_size(kBlockSize, m)
    };

    // Запуск ядра
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "linear_fwd_kern", ([&] {
        linear_fwd_kern<scalar_t><<<dimGrid, dimBlock>>>(
            input.packed_accessor32<scalar_t, 2>(),
            weight.packed_accessor32<scalar_t, 2>(),
            bias.packed_accessor32<scalar_t, 1>(),
            output.packed_accessor32<scalar_t, 2>()
        );
    }));

    return output;
}

std::vector<torch::Tensor> backward_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor d_output) {
    // Проверка входных данных
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(d_output);

    // Проверка размерностей
    TORCH_CHECK(input.size(1) == weight.size(1), "Input and weight dimensions do not match");
    TORCH_CHECK(weight.size(0) == bias.size(0), "Weight and bias dimensions do not match");
    TORCH_CHECK(d_output.size(0) == input.size(0), "d_output and input batch sizes do not match");
    TORCH_CHECK(d_output.size(1) == weight.size(0), "d_output and weight output dimensions do not match");

    // Инициализация градиентов
    torch::Tensor d_input = torch::zeros_like(input);
    torch::Tensor d_weight = torch::zeros_like(weight);
    torch::Tensor d_bias = torch::zeros_like(bias);

    // Размеры блоков и сетки
    constexpr dim3 dimBlock = {kBlockSize, kBlockSize};
    const dim3 dimGrid = {
        calc_grid_size(kBlockSize, weight.size(1)),
        calc_grid_size(kBlockSize, input.size(0))
    };

    // Запуск ядра
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "linear_bwd_kern", ([&] {
        linear_bwd_kern<scalar_t><<<dimGrid, dimBlock>>>(
            input.packed_accessor32<scalar_t, 2>(),
            weight.packed_accessor32<scalar_t, 2>(),
            d_output.packed_accessor32<scalar_t, 2>(),
            d_input.packed_accessor32<scalar_t, 2>(),
            d_weight.packed_accessor32<scalar_t, 2>(),
            d_bias.packed_accessor32<scalar_t, 1>()
        );
    }));

    return {d_input, d_weight, d_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_forward_linear", &forward_linear, "Custom forward linear layer");
    m.def("my_backward_linear", &backward_linear, "Custom backward linear layer");
}
