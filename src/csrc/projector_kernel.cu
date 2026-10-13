#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// --------------------------------------------------------
// 1. CUDA Kernels (Joseph's Ray-Casting Algorithm Skeleton)
// --------------------------------------------------------

__global__ void forward_project_kernel(
    const float* __restrict__ volume,
    const float* __restrict__ angles,
    float* __restrict__ sinogram,
    int batch_size, int channels, int depth, int height, int width,
    int num_angles, int det_u, int det_v) 
{
    // Total number of rays: Batch * Channels * Angles * DetectorWidth * DetectorHeight
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rays = batch_size * channels * num_angles * det_u * det_v;

    if (idx < total_rays) {
        // TODO: Decode 1D idx into (batch, channel, angle, u, v)
        // TODO: Compute source and detector 3D coordinates based on angles[angle]
        // TODO: Implement Joseph's method: determine principal ray axis (X, Y, or Z)
        // TODO: Traverse volume grid boundaries and linearly interpolate intensities
        
        // Initialize sinogram pixel to 0
        sinogram[idx] = 0.0f; 
    }
}

__global__ void backward_project_kernel(
    const float* __restrict__ grad_sinogram,
    const float* __restrict__ angles,
    float* __restrict__ grad_volume,
    int batch_size, int channels, int depth, int height, int width,
    int num_angles, int det_u, int det_v) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rays = batch_size * channels * num_angles * det_u * det_v;

    if (idx < total_rays) {
        // TODO: Decode 1D idx into (batch, channel, angle, u, v)
        // TODO: Retrace the exact ray path calculated in the forward pass
        // TODO: Distribute grad_sinogram[idx] back to the nearest voxels using atomicAdd()
        
        float grad_val = grad_sinogram[idx];
        // Example atomic addition mapping back to a 1D volume index:
        // atomicAdd(&grad_volume[voxel_idx], grad_val * interpolation_weight);
    }
}

// --------------------------------------------------------
// 2. C++ Host Interface (Dispatches to CUDA)
// --------------------------------------------------------

// Helper to calculate grid/block sizes
inline int get_blocks(const int threads, const int total_elements) {
    return (total_elements + threads - 1) / threads;
}

torch::Tensor forward_project_cuda(torch::Tensor volume, torch::Tensor angles) {
    // Extract dimensions
    const int batch_size = volume.size(0);
    const int channels = volume.size(1);
    const int depth = volume.size(2);
    const int height = volume.size(3);
    const int width = volume.size(4);
    const int num_angles = angles.size(0);
    
    // Define detector dimensions (assuming 2D flat panel detector for 3D volume)
    // In practice, derive these based on system geometry configurations
    const int det_u = width; 
    const int det_v = depth;

    // Allocate output tensor: [Batch, Channels, Angles, det_v, det_u]
    auto sinogram = torch::zeros({batch_size, channels, num_angles, det_v, det_u}, volume.options());

    const int total_rays = batch_size * channels * num_angles * det_u * det_v;
    const int threads = 256;
    const int blocks = get_blocks(threads, total_rays);

    // Launch Kernel
    forward_project_kernel<<<blocks, threads>>>(
        volume.data_ptr<float>(),
        angles.data_ptr<float>(),
        sinogram.data_ptr<float>(),
        batch_size, channels, depth, height, width,
        num_angles, det_u, det_v
    );

    // Synchronize and check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA forward kernel failed: ", cudaGetErrorString(err));

    return sinogram;
}

torch::Tensor backward_project_cuda(torch::Tensor grad_sinogram, torch::Tensor angles, std::vector<int64_t> volume_shape) {
    // Extract dimensions
    const int batch_size = volume_shape[0];
    const int channels = volume_shape[1];
    const int depth = volume_shape[2];
    const int height = volume_shape[3];
    const int width = volume_shape[4];
    
    const int num_angles = angles.size(0);
    const int det_v = grad_sinogram.size(3);
    const int det_u = grad_sinogram.size(4);

    // Allocate gradient volume initialized to 0
    auto options = grad_sinogram.options();
    auto grad_volume = torch::zeros({batch_size, channels, depth, height, width}, options);

    const int total_rays = batch_size * channels * num_angles * det_u * det_v;
    const int threads = 256;
    const int blocks = get_blocks(threads, total_rays);

    // Launch Kernel
    backward_project_kernel<<<blocks, threads>>>(
        grad_sinogram.data_ptr<float>(),
        angles.data_ptr<float>(),
        grad_volume.data_ptr<float>(),
        batch_size, channels, depth, height, width,
        num_angles, det_u, det_v
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA backward kernel failed: ", cudaGetErrorString(err));

    return grad_volume;
}