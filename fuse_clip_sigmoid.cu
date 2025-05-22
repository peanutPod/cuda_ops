#include<cuda_runtime.h>
#include<cuda_fp16.h>


// 使用warp-level操作进行better数据重用
template <typename T>
__global__ void fuse_clip_sigmoid_kernel_warp_opt(const T* input0, T* output, const int size) 
{

    const half zero = __float2half(0.0f);
    const half one = __float2half(1.0f);
    const half min_clip_val = __float2half(0.000000999f);
    const half ln=__float2half(0.693147181f);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_offset = (blockIdx.x * (blockDim.x / 32) + warp_id) * 32;
    int global_idx = warp_offset + lane_id;
    
    if (global_idx < size) {
        T x = input0[global_idx];
        
        // 使用快速数学库函数
        x = __hgt(x, zero) ? x : zero;
        x = __hlt(x, one) ? x : one;
        
        T sub_result = __hsub(one, x);
        
        T left_clip = __hgt(x, min_clip_val) ? x : min_clip_val;
        T right_clip = __hgt(sub_result, min_clip_val) ? sub_result : min_clip_val;
        
        // 使用更高效的division-by-multiplication技术
        T div_result = __hdiv(left_clip, right_clip);
        
        // 使用更精确的log近似
        T log_result = hlog2(div_result) * ln;
        
        output[global_idx] = log_result;
    }
}


template <typename T>
__global__ void fuse_clip_sigmoid_kernel_optimized(const T* input0, T* output, const int size)
{
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    // if (index < size)
    // {
    //     // 直接使用T类型，避免不必要的转换
    //     T x = input0[index];
    //     const half zero = __float2half(0.0f);
    //     const half one = __float2half(1.0f);
    //     const half min_clip_val = __float2half(0.000000999f);
    //     const half ln=__float2half(0.693147181f);

    //     // 使用预计算的常量
    //     x = __hgt(x, zero) ? x : zero;
    //     x = __hlt(x, one) ? x : one;
        
    //     T sub_result = __hsub(__float2half(1.0f), x);
    //     T left_clip = __hgt(x, min_clip_val) ? x : min_clip_val;
    //     T right_clip = __hgt(sub_result, min_clip_val) ? sub_result : min_clip_val;
        
    //     T div_result = __hdiv(left_clip, right_clip);
    //     T log_result = hlog2(div_result) * ln;
        
    //     output[index] = log_result;
    // }
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        // 直接使用T类型，避免不必要的转换
        T x = input0[index];

        // 使用预计算的常量
        x = __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
        x = __hlt(x, __float2half(1.0f)) ? x : __float2half(1.0f);
        
        T sub_result = __hsub(__float2half(1.0f), x);
        T left_clip = __hgt(x, __float2half(0.000000999f)) ? x : __float2half(0.000000999f);
        T right_clip = __hgt(sub_result, __float2half(0.000000999f)) ? sub_result : __float2half(0.000000999f);
        
        T div_result = __hdiv(left_clip, right_clip);
        T log_result = hlog2(div_result) * __float2half(0.693147181f);
        
        output[index] = log_result;
    }


}

// Original implementation for reference
template <typename T>
__global__ void fuse_clip_sigmoid_kernel(const T* input0, T* output, const int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        const half* input0_ptr = (const half*)input0;
        half* output_ptr = (half*)output;

        // Get input value
        half x = input0_ptr[index];
        
        // First clip: min=0
        x = __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
        
        // Second clip: max=1
        x = __hlt(x, __float2half(1.0f)) ? x : __float2half(1.0f);
        
        // Sub operation: A=1
        half sub_result = __hsub(__float2half(1.0f), x);
        
        // Two more clip operations with min=0.000000999
        const half min_clip_val = __float2half(0.000000999f);
        half left_clip = __hgt(x, min_clip_val) ? x : min_clip_val;
        half right_clip = __hgt(sub_result, min_clip_val) ? sub_result : min_clip_val;
        
        // Division
        half div_result = __hdiv(left_clip,right_clip);
        
        // Log operation
        half log_result;
        // Approximation for log using native functions
        log_result = hlog2(div_result) * __float2half(0.693147181f);
        
        // Perform the multiplication with input1
        output_ptr[index] = log_result;
    }
}

// Half2 optimized implementation
__global__ void fuse_clip_sigmoid_kernel_half2(const half2* input0, half2* output, const int size_half2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size_half2)
    {
        // Load 2 elements at once
        half2 x = input0[index];
        
        // First clip: min=0
        const half2 zero = __float2half2_rn(0.0f);
        // Use proper half2 comparison and selection
        x = __hfma2(__hgt2(x, zero), x, __hmul2(__hle2(x, zero), zero));
        
        // Second clip: max=1
        const half2 one = __float2half2_rn(1.0f);
        // Use proper half2 comparison and selection
        x = __hfma2(__hlt2(x, one), x, __hmul2(__hge2(x, one), one));
        
        // Sub operation: A=1
        half2 sub_result = __hsub2(one, x);
        
        // Two more clip operations with min=0.000000999
        const half2 min_clip_val = __float2half2_rn(0.000000999f);
        // Use proper half2 comparison and selection
        half2 left_clip = __hfma2(__hgt2(x, min_clip_val), x, __hmul2(__hle2(x, min_clip_val), min_clip_val));
        half2 right_clip = __hfma2(__hgt2(sub_result, min_clip_val), sub_result, __hmul2(__hle2(sub_result, min_clip_val), min_clip_val));
        
        // Division
        half2 div_result = __h2div(left_clip, right_clip);
        
        // Log operation
        // Natural log approximation: ln(x) = log2(x) * ln(2)
        const half2 ln2 = __float2half2_rn(0.693147181f);
        half2 log_result = h2log2(div_result) * ln2;
        
        // Store result - process 2 elements at once
        output[index] = log_result;
    }
}


extern "C" void launch_fuse_clip_sigmoid2(const half* input0, half* output, 
                                        int size, cudaStream_t stream = 0)
{
    const int block_size = 128;
    
    // Calculate half2 size (half the original size, rounded up)
    int size_half2 = (size + 1) / 2;
    const int grid_size = (size_half2 + block_size - 1) / block_size;
    
    // Cast pointers to half2* for vectorized processing
    fuse_clip_sigmoid_kernel_half2<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const half2*>(input0), 
        reinterpret_cast<half2*>(output), 
        size);
}



#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) 
                  << "(" << cudaGetErrorString(result) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}



// CPU reference implementation to match the CUDA kernel functionality
void cpu_reference_implementation(const float* input0, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        // Get input value
        float x = input0[i];
        
        // First clip: min=0
        x = (x > 0.0f) ? x : 0.0f;
        
        // Second clip: max=1
        x = (x < 1.0f) ? x : 1.0f;
        
        // Sub operation: A=1
        float sub_result = 1.0f - x;
        
        // Two more clip operations with min=0.000000999
        const float min_clip_val = 0.000000999f;
        float left_clip = (x > min_clip_val) ? x : min_clip_val;
        float right_clip = (sub_result > min_clip_val) ? sub_result : min_clip_val;
        
        // Division
        float div_result = left_clip /right_clip;
        
        // Log operation (using natural logarithm)
        float log_result = std::log(div_result);
        
        // Store result
        output[i] = log_result;
    }
}


// Custom launch function for the original scalar kernel
void launch_scalar_kernel(const half* input0, half* output, int size, cudaStream_t stream = 0) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    fuse_clip_sigmoid_kernel_optimized<half><<<grid_size, block_size, 0, stream>>>(
        input0, output, size);
}





int main() {
    // Test parameters
    const int batch = 1;
    const int height = 100;
    const int width = 40;
    const int size = batch * height * width;
    const int num_iterations = 10000; // For timing
    
    std::cout << "Testing CUDA kernels with shape (" 
              << batch << ", " << height << ", " << width << ")" << std::endl;
    
    // Allocate host memory
    std::vector<float> h_input0_float(size);
    std::vector<float> h_output_float_scalar(size);
    std::vector<float> h_output_float_half2(size);
    std::vector<float> h_ref_output(size);
    
    std::vector<half> h_input0(size);
    std::vector<half> h_output_scalar(size);
    std::vector<half> h_output_half2(size);
    
    // Initialize input with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        h_input0_float[i] = dist(gen);
        h_input0[i] = __float2half(h_input0_float[i]);
    }
    
    // Allocate device memory
    half *d_input0, *d_output_scalar, *d_output_half2;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input0, size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_scalar, size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_half2, size * sizeof(half)));
    
    // Copy input data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input0, h_input0.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Warmup runs
    launch_scalar_kernel(d_input0, d_output_scalar, size, 0);
    launch_fuse_clip_sigmoid2(d_input0, d_output_half2, size, 0);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // === 1. Measure scalar kernel performance ===
    std::cout << "\n===== Testing Original Scalar Kernel =====" << std::endl;
    float total_time_scalar = 0.0f;
    
    for (int i = 0; i < num_iterations; ++i) {
        float milliseconds = 0.0f;
        
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        launch_scalar_kernel(d_input0, d_output_scalar, size, 0);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time_scalar += milliseconds;
    }
    
    float avg_time_scalar = total_time_scalar / num_iterations;
    std::cout << "Average execution time: " << avg_time_scalar << " ms" << std::endl;
    
    // === 2. Measure half2 kernel performance ===
    std::cout << "\n===== Testing Half2 Vectorized Kernel =====" << std::endl;
    float total_time_half2 = 0.0f;
    
    for (int i = 0; i < num_iterations; ++i) {
        float milliseconds = 0.0f;
        
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        launch_fuse_clip_sigmoid2(d_input0, d_output_half2, size, 0);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time_half2 += milliseconds;
    }
    
    float avg_time_half2 = total_time_half2 / num_iterations;
    std::cout << "Average execution time: " << avg_time_half2 << " ms" << std::endl;
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_scalar.data(), d_output_scalar, size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_half2.data(), d_output_half2, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Convert GPU outputs to float for comparison
    for (int i = 0; i < size; ++i) {
        h_output_float_scalar[i] = __half2float(h_output_scalar[i]);
        h_output_float_half2[i] = __half2float(h_output_half2[i]);
    }
    
    // Run CPU reference implementation
    cpu_reference_implementation(h_input0_float.data(), h_ref_output.data(), size);
    
    // Compare GPU and CPU results
    float max_diff_scalar = 0.0f;
    float avg_diff_scalar = 0.0f;
    float max_rel_diff_scalar = 0.0f;
    
    float max_diff_half2 = 0.0f;
    float avg_diff_half2 = 0.0f;
    float max_rel_diff_half2 = 0.0f;
    
    float max_diff_between = 0.0f;
    float avg_diff_between = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        // Compare scalar with reference
        float diff_scalar = std::abs(h_output_float_scalar[i] - h_ref_output[i]);
        avg_diff_scalar += diff_scalar;
        max_diff_scalar = std::max(max_diff_scalar, diff_scalar);
        
        if (std::abs(h_ref_output[i]) > 1e-6f) {
            float rel_diff = diff_scalar / std::abs(h_ref_output[i]);
            max_rel_diff_scalar = std::max(max_rel_diff_scalar, rel_diff);
        }
        
        // Compare half2 with reference
        float diff_half2 = std::abs(h_output_float_half2[i] - h_ref_output[i]);
        avg_diff_half2 += diff_half2;
        max_diff_half2 = std::max(max_diff_half2, diff_half2);
        
        if (std::abs(h_ref_output[i]) > 1e-6f) {
            float rel_diff = diff_half2 / std::abs(h_ref_output[i]);
            max_rel_diff_half2 = std::max(max_rel_diff_half2, rel_diff);
        }
        
        // Compare scalar with half2
        float diff_between = std::abs(h_output_float_scalar[i] - h_output_float_half2[i]);
        avg_diff_between += diff_between;
        max_diff_between = std::max(max_diff_between, diff_between);
    }
    
    avg_diff_scalar /= size;
    avg_diff_half2 /= size;
    avg_diff_between /= size;
    
    // Print summary performance results
    std::cout << "\n=================== Performance Summary ===================" << std::endl;
    std::cout << "Scalar kernel execution time: " << avg_time_scalar << " ms" << std::endl;
    std::cout << "Half2 kernel execution time:  " << avg_time_half2 << " ms" << std::endl;
    std::cout << "Speedup: " << avg_time_scalar / avg_time_half2 << "x" << std::endl;
    
    // Print accuracy results
    std::cout << "\n=================== Accuracy Results ===================" << std::endl;
    
    std::cout << "                       | Scalar Kernel | Half2 Kernel" << std::endl;
    std::cout << "-----------------------|--------------|-------------" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Avg absolute diff      | " << std::setw(12) << avg_diff_scalar 
              << " | " << std::setw(12) << avg_diff_half2 << std::endl;
    std::cout << "Max absolute diff      | " << std::setw(12) << max_diff_scalar 
              << " | " << std::setw(12) << max_diff_half2 << std::endl;
    std::cout << "Max relative diff (%)  | " << std::setw(12) << max_rel_diff_scalar * 100.0f 
              << " | " << std::setw(12) << max_rel_diff_half2 * 100.0f << std::endl;
    
    // Print difference between implementations
    std::cout << "\nDifference between scalar and half2 implementations:" << std::endl;
    std::cout << "  Average absolute difference: " << avg_diff_between << std::endl;
    std::cout << "  Maximum absolute difference: " << max_diff_between << std::endl;
    
    // Print sample outputs for visual inspection
    std::cout << "\n=================== Sample Values ===================" << std::endl;
    std::cout << std::setw(7) << "Index" << " | " 
              << std::setw(12) << "Input" << " | "
              << std::setw(12) << "Scalar" << " | " 
              << std::setw(12) << "Half2" << " | " 
              << std::setw(12) << "CPU Ref" << std::endl;
    std::cout << "-------|--------------|--------------|--------------|-------------" << std::endl;
    
    for (int i = 0; i < std::min(5, size); ++i) {
        std::cout << std::setw(7) << i << " | "
                  << std::setw(12) << h_input0_float[i] << " | "
                  << std::setw(12) << h_output_float_scalar[i] << " | "
                  << std::setw(12) << h_output_float_half2[i] << " | "
                  << std::setw(12) << h_ref_output[i] << std::endl;
    }
    
    // Clean up
    CHECK_CUDA_ERROR(cudaFree(d_input0));
    CHECK_CUDA_ERROR(cudaFree(d_output_scalar));
    CHECK_CUDA_ERROR(cudaFree(d_output_half2));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return 0;
}