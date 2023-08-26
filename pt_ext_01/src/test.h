/*test.h*/
#include <torch/extension.h>
#include <vector>

// forward propagation
torch::Tensor Test_forward_cpu(const torch::Tensor& inputA, const torch::Tensor& inputB);
// backward propagation
std::vector<torch::Tensor> Test_backward_cpu(const torch::Tensor& gradOutput);