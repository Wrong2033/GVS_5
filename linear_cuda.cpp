#include <torch/extension.h>
#include <vector>
#include <iostream>

// Объявление функции прямого прохода из CUDA-файла
torch::Tensor linear_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias);

// Класс пользовательского линейного слоя
class LinearCUDA : public torch::nn::Module {
public:
    LinearCUDA(int64_t in_features, int64_t out_features) {
        // Инициализация весов и смещения на GPU
        weight_ = register_parameter("weight", 
            torch::empty({out_features, in_features}, torch::device(torch::kCUDA)));
        bias_ = register_parameter("bias", 
            torch::empty(out_features, torch::device(torch::kCUDA)));
        
        // Инициализация параметров как в PyTorch
        torch::nn::init::kaiming_uniform_(weight_, sqrt(5.0));
        if (bias_.defined()) {
            auto fan_in = weight_.size(1);
            auto bound = 1.0 / sqrt(fan_in);
            torch::nn::init::uniform_(bias_, -bound, bound);
        }
    }

    torch::Tensor forward(const torch::Tensor& input) {
        return linear_forward_cuda(input, weight_, bias_);
    }

private:
    torch::Tensor weight_, bias_;
};

// Регистрация модуля для PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<LinearCUDA>(m, "LinearCUDA")
        .def(py::init<int64_t, int64_t>())
        .def("forward", &LinearCUDA::forward);
    m.def("linear_forward_cuda", &linear_forward_cuda, "Linear forward (CUDA)");
}
