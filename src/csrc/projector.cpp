#include <torch/extension.h>
#include <vector>

// Forward declarations of the CUDA functions implemented in projector_kernel.cu
torch::Tensor forward_project_cuda(torch::Tensor volume, torch::Tensor angles);
torch::Tensor backward_project_cuda(torch::Tensor grad_sinogram, torch::Tensor angles, std::vector<int64_t> volume_shape);

// Helper macros to ensure tensors are formatted correctly for CUDA memory
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor forward_project(torch::Tensor volume, torch::Tensor angles) {
    CHECK_INPUT(volume);
    CHECK_INPUT(angles);
    return forward_project_cuda(volume, angles);
}

torch::Tensor backward_project(torch::Tensor grad_sinogram, torch::Tensor angles, std::vector<int64_t> volume_shape) {
    CHECK_INPUT(grad_sinogram);
    CHECK_INPUT(angles);
    return backward_project_cuda(grad_sinogram, angles, volume_shape);
}

// Binds the C++ functions to the Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_project, "Forward Projection (CUDA)");
    m.def("backward", &backward_project, "Backward Projection (CUDA)");
}