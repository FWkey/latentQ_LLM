#include <torch/extension.h>

void launch_quant(at::Tensor& src, at::Tensor& tgt,
               const int k1, const float b1, const int k2, const float b2, const int k3, int n);

void launch_dequant(at::Tensor& src, at::Tensor& tgt,
               const int k1, const float b1, const int k2, const float b2, const int k3, int n);

void launch_quant_linear(torch::Tensor& src, torch::Tensor& tgt, int n);
void launch_dequant_linear(torch::Tensor& src, torch::Tensor& tgt, int n);
void launch_quant_linear_s(torch::Tensor& src, torch::Tensor& tgt, int n);

void our_quant(at::Tensor fp_input, at::Tensor int_input, const int k1, const float b1, const int k2, const float b2, const int k3) {
  int n = at::numel(int_input);
  launch_quant(fp_input, int_input, k1, b1, k2, b2, k3, n);
}

void our_dequant(at::Tensor int_input, at::Tensor fp_output, const int k1, const float b1, const int k2, const float b2, const int k3) {
  int n = at::numel(int_input);
  launch_dequant(int_input, fp_output, k1, b1, k2, b2, k3, n);
}

void linear_quant(at::Tensor fp_input, at::Tensor int_input) {
  int n = at::numel(int_input);
  launch_quant_linear(fp_input, int_input, n);
}

void linear_dequant(at::Tensor int_input, at::Tensor fp_output) {
  int n = at::numel(int_input);
  launch_dequant_linear(int_input, fp_output, n);
}

void linear_quant_s(at::Tensor fp_input, at::Tensor int_input) {
  int n = at::numel(int_input);
  launch_quant_linear_s(fp_input, int_input, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant",
          &our_quant,
          "multi_random_quant kernel warpper");
    m.def("dequant",
          &our_dequant,
          "multi_random_dequant kernel warpper");
    m.def("uniform_quant",
          &linear_quant,
          "uniform_random_quant kernel warpper");
    m.def("uniform_quant_s",
          &linear_quant_s,
          "uniform_determined_quant kernel warpper");
    m.def("uniform_dequant",
          &linear_dequant,
          "uniform_random_dequant kernel warpper");
}