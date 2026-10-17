#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// System geometry constants (can be passed dynamically from Python later)
#define SAD 500.0f  // Source-to-Axis Distance
#define SDD 1000.0f // Source-to-Detector Distance

__device__ inline bool is_inside(int x, int y, int z, int w, int h, int d) {
    return (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d);
}

// --------------------------------------------------------
// Forward Projection (Ray to Volume Integration)
// --------------------------------------------------------
__global__ void forward_project_kernel(
    const float* __restrict__ volume,
    const float* __restrict__ angles,
    float* __restrict__ sinogram,
    int batch_size, int channels, int depth, int height, int width,
    int num_angles, int det_u, int det_v) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rays = batch_size * channels * num_angles * det_u * det_v;

    if (idx >= total_rays) return;

    // Decode 1D thread index into grid coordinates
    int u = idx % det_u;
    int v = (idx / det_u) % det_v;
    int a = (idx / (det_u * det_v)) % num_angles;
    int c = (idx / (det_u * det_v * num_angles)) % channels;
    int b = (idx / (det_u * det_v * num_angles * channels));

    float theta = angles[a];
    
    // Calculate Source (S) and Detector (D) coordinates in 3D space
    float su = u - det_u / 2.0f;
    float sv = v - det_v / 2.0f;
    
    float source_x = SAD * cosf(theta);
    float source_y = SAD * sinf(theta);
    float source_z = 0.0f;
    
    float det_x = -SDD * cosf(theta) - su * sinf(theta);
    float det_y = -SDD * sinf(theta) + su * cosf(theta);
    float det_z = sv;

    // Ray direction vector
    float rx = det_x - source_x;
    float ry = det_y - source_y;
    float rz = det_z - source_z;
    
    // Normalize ray direction
    float length = sqrtf(rx*rx + ry*ry + rz*rz);
    rx /= length; ry /= length; rz /= length;

    // Joseph's Method: Identify the driving axis (the axis with the largest progression)
    int driving_axis = 0; // 0 for X, 1 for Y
    if (fabsf(ry) > fabsf(rx)) driving_axis = 1;

    float ray_sum = 0.0f;
    
    // Step size normalizations based on the driving axis
    float step_x = rx / fabsf(driving_axis == 0 ? rx : ry);
    float step_y = ry / fabsf(driving_axis == 0 ? rx : ry);
    float step_z = rz / fabsf(driving_axis == 0 ? rx : ry);

    // Center the volume grid at (0,0,0)
    float half_w = width / 2.0f;
    float half_h = height / 2.0f;
    float half_d = depth / 2.0f;

    // Traverse the volume grid along the driving axis
    int grid_steps = (driving_axis == 0) ? width : height;
    
    for (int step = 0; step < grid_steps; ++step) {
        // Calculate intersection point on the current perpendicular plane
        float current_t = (driving_axis == 0) ? (step - half_w - source_x) / rx : (step - half_h - source_y) / ry;
        
        float px = source_x + current_t * rx + half_w;
        float py = source_y + current_t * ry + half_h;
        float pz = source_z + current_t * rz + half_d;

        // Determine 4 nearest voxels on the non-driving plane for bilinear interpolation
        int ix = floorf(px);
        int iy = floorf(py);
        int iz = floorf(pz);

        float wx = px - ix;
        float wy = py - iy;
        float wz = pz - iz;

        // Sample the volume if inside bounds
        int vol_offset = ((b * channels + c) * depth * height * width);
        
        if (driving_axis == 0) { // X is driving, interpolate Y and Z
            if (is_inside(step, iy, iz, width, height, depth)) 
                ray_sum += volume[vol_offset + iz * height * width + iy * width + step] * (1-wy) * (1-wz);
            if (is_inside(step, iy+1, iz, width, height, depth)) 
                ray_sum += volume[vol_offset + iz * height * width + (iy+1) * width + step] * (wy) * (1-wz);
            if (is_inside(step, iy, iz+1, width, height, depth)) 
                ray_sum += volume[vol_offset + (iz+1) * height * width + iy * width + step] * (1-wy) * (wz);
            if (is_inside(step, iy+1, iz+1, width, height, depth)) 
                ray_sum += volume[vol_offset + (iz+1) * height * width + (iy+1) * width + step] * (wy) * (wz);
        } else { // Y is driving, interpolate X and Z
            if (is_inside(ix, step, iz, width, height, depth)) 
                ray_sum += volume[vol_offset + iz * height * width + step * width + ix] * (1-wx) * (1-wz);
            if (is_inside(ix+1, step, iz, width, height, depth)) 
                ray_sum += volume[vol_offset + iz * height * width + step * width + (ix+1)] * (wx) * (1-wz);
            if (is_inside(ix, step, iz+1, width, height, depth)) 
                ray_sum += volume[vol_offset + (iz+1) * height * width + step * width + ix] * (1-wx) * (wz);
            if (is_inside(ix+1, step, iz+1, width, height, depth)) 
                ray_sum += volume[vol_offset + (iz+1) * height * width + step * width + (ix+1)] * (wx) * (wz);
        }
    }
    
    // Scale by voxel intersection length correction
    float correction = sqrtf(1.0f + (step_y*step_y) + (step_z*step_z));
    sinogram[idx] = ray_sum * correction;
}

// --------------------------------------------------------
// Backward Projection (Gradient Distribution)
// --------------------------------------------------------
__global__ void backward_project_kernel(
    const float* __restrict__ grad_sinogram,
    const float* __restrict__ angles,
    float* __restrict__ grad_volume,
    int batch_size, int channels, int depth, int height, int width,
    int num_angles, int det_u, int det_v) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rays = batch_size * channels * num_angles * det_u * det_v;

    if (idx >= total_rays) return;

    int u = idx % det_u;
    int v = (idx / det_u) % det_v;
    int a = (idx / (det_u * det_v)) % num_angles;
    int c = (idx / (det_u * det_v * num_angles)) % channels;
    int b = (idx / (det_u * det_v * num_angles * channels));

    float theta = angles[a];
    
    // Same ray parameterization as forward pass to guarantee physical consistency
    float su = u - det_u / 2.0f;
    float sv = v - det_v / 2.0f;
    
    float source_x = SAD * cosf(theta);
    float source_y = SAD * sinf(theta);
    float source_z = 0.0f;
    
    float det_x = -SDD * cosf(theta) - su * sinf(theta);
    float det_y = -SDD * sinf(theta) + su * cosf(theta);
    float det_z = sv;

    float rx = det_x - source_x;
    float ry = det_y - source_y;
    float rz = det_z - source_z;
    
    float length = sqrtf(rx*rx + ry*ry + rz*rz);
    rx /= length; ry /= length; rz /= length;

    int driving_axis = 0; 
    if (fabsf(ry) > fabsf(rx)) driving_axis = 1;

    float step_x = rx / fabsf(driving_axis == 0 ? rx : ry);
    float step_y = ry / fabsf(driving_axis == 0 ? rx : ry);
    float step_z = rz / fabsf(driving_axis == 0 ? rx : ry);

    float half_w = width / 2.0f;
    float half_h = height / 2.0f;
    float half_d = depth / 2.0f;

    float correction = sqrtf(1.0f + (step_y*step_y) + (step_z*step_z));
    float grad_val = grad_sinogram[idx] * correction;

    int grid_steps = (driving_axis == 0) ? width : height;
    
    for (int step = 0; step < grid_steps; ++step) {
        float current_t = (driving_axis == 0) ? (step - half_w - source_x) / rx : (step - half_h - source_y) / ry;
        
        float px = source_x + current_t * rx + half_w;
        float py = source_y + current_t * ry + half_h;
        float pz = source_z + current_t * rz + half_d;

        int ix = floorf(px);
        int iy = floorf(py);
        int iz = floorf(pz);

        float wx = px - ix;
        float wy = py - iy;
        float wz = pz - iz;

        int vol_offset = ((b * channels + c) * depth * height * width);
        
        // atomicAdd ensures race-condition safety as multiple rays distribute gradients to the same voxels
        if (driving_axis == 0) {
            if (is_inside(step, iy, iz, width, height, depth)) 
                atomicAdd(&grad_volume[vol_offset + iz * height * width + iy * width + step], grad_val * (1-wy) * (1-wz));
            if (is_inside(step, iy+1, iz, width, height, depth)) 
                atomicAdd(&grad_volume[vol_offset + iz * height * width + (iy+1) * width + step], grad_val * (wy) * (1-wz));
            if (is_inside(step, iy, iz+1, width, height, depth)) 
                atomicAdd(&grad_volume[vol_offset + (iz+1) * height * width + iy * width + step], grad_val * (1-wy) * (wz));
            if (is_inside(step, iy+1, iz+1, width, height, depth)) 
                atomicAdd(&grad_volume[vol_offset + (iz+1) * height * width + (iy+1) * width + step], grad_val * (wy) * (wz));
        } else {
            if (is_inside(ix, step, iz, width, height, depth)) 
                atomicAdd(&grad_volume[vol_offset + iz * height * width + step * width + ix], grad_val * (1-wx) * (1-wz));
            if (is_inside(ix+1, step, iz, width, height, depth)) 
                atomicAdd(&grad_volume[vol_offset + iz * height * width + step * width + (ix+1)], grad_val * (wx) * (1-wz));
            if (is_inside(ix, step, iz+1, width, height, depth)) 
                atomicAdd(&grad_volume[vol_offset + (iz+1) * height * width + step * width + ix], grad_val * (1-wx) * (wz));
            if (is_inside(ix+1, step, iz+1, width, height, depth)) 
                atomicAdd(&grad_volume[vol_offset + (iz+1) * height * width + step * width + (ix+1)], grad_val * (wx) * (wz));
        }
    }
}

// --------------------------------------------------------
// C++ Host Interface (Dispatches to CUDA)
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