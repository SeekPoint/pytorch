作者：旅行大喵
链接：https://www.zhihu.com/question/354886894/answer/888033350
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#include <torch/extension.h>

void gelu_fwd_interface(float*, float*, int64_t);
void gelu_bwd_interface(float*, float*, float*, int64_t);

torch::Tensor gelu_fwd(torch::Tensor input) {
    if(input.device().type() == torch::kCPU) {
        return input*torch::sigmoid(1.702*input);
    } else if (input.device().type() == torch::kCUDA){
        TORCH_CHECK(input.dtype() == torch::kFloat32,
            "Datatype not implemented");
        auto ret = torch::zeros_like(input);
        int64_t size = ret.numel();
        gelu_fwd_interface(input.data_ptr<float>(),
                           ret.data_ptr<float>(), size);
        return ret;
    }
    AT_ERROR("No such device: ", input.device());
}

torch::Tensor gelu_bwd(torch::Tensor grad_out, torch::Tensor input) {
    if(input.device().type() == torch::kCPU) {
        auto tmp = torch::sigmoid(1.702*input);
        return grad_out*(tmp+1.702*input*tmp*(1.0-tmp));
    } else if (input.device().type() == torch::kCUDA){
        TORCH_CHECK(input.dtype() == torch::kFloat32,
            "Datatype not implemented");
        TORCH_CHECK(grad_out.dtype() == torch::kFloat32,
            "Datatype not implemented");
        auto ret = torch::zeros_like(input);
        int64_t size = ret.numel();
        gelu_bwd_interface(grad_out.data_ptr<float>(),
                           input.data_ptr<float>(),
                           ret.data_ptr<float>(), size);
        return ret;
    }
    AT_ERROR("No such device: ", input.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gelu_fwd, "GELU forward");
  m.def("backward", &gelu_bwd, "GELU backward");
}