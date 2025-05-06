#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
// Clip operator limits the given input within an interval. 
// The interval is specified by the inputs 'min' and 'max'. 
// They default to numeric_limits::lowest() and numeric_limits::max(), respectively.
// When 'min' is greater than 'max', the clip operator sets all the 'input' values to the value of 'max'. 
// Thus, this is equivalent to 'Min(max, Max(input, min))'.

__global__ void clip_kernel(const float* input,float* output, float min ,float max, int size){
    int idx =blockIdx.x* blockDim.x + threadIdx.x;
    if (idx < size) {
        float data = input[idx];
        output[idx] = fminf(fmaxf(data, min), max);
    }
}

__global__ void clip_kernel_half(const __half* input, __half* output, const __half min, const __half max, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        #if   __CUDA_ARCH__ >= 530
        __half data = input[idx];
        output[idx] = __hmin(__hmax(data, min), max);
        #else
        // Fallback to single-precision arithmetic
        float data_f = __half2float(input[idx]);
        float min_f = __half2float(min);
        float max_f = __half2float(max);
        float result_f = fminf(fmaxf(data_f, min_f), max_f);
        output[idx] = __float2half(result_f);
        #endif
    }
}

// Test function for single-precision (float)
void test_clip_float(const float* h_input, float* h_output, float min, float max, int size) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    clip_kernel<<<blocks, threads>>>(d_input, d_output, min, max, size);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    clip_kernel<<<blocks, threads>>>(d_input, d_output, min, max, size);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for clip_kernel (float): %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Test function for half-precision (__half)
void test_clip_half(const float* h_input_float, float* h_output_float, float min, float max, int size) {
    __half *h_input_half = (__half*)malloc(size * sizeof(__half));
    __half *h_output_half = (__half*)malloc(size * sizeof(__half));

    // Convert input data from float to __half
    for (int i = 0; i < size; i++) {
        h_input_half[i] = __float2half(h_input_float[i]);
    }

    __half *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(__half));
    cudaMalloc(&d_output, size * sizeof(__half));


    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Copy input data to device
    cudaMemcpy(d_input, h_input_half, size * sizeof(__half), cudaMemcpyHostToDevice);
    
    clip_kernel_half<<<blocks, threads>>>(d_input, d_output, __float2half(min), __float2half(max), size);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    clip_kernel_half<<<blocks, threads>>>(d_input, d_output, __float2half(min), __float2half(max), size);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for clip_kernel_half (__half): %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output_half, d_output, size * sizeof(__half), cudaMemcpyDeviceToHost);

    // Convert output data from __half to float
    for (int i = 0; i < size; i++) {
        h_output_float[i] = __half2float(h_output_half[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input_half);
    free(h_output_half);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <typename T>
__global__ void clip_kernel_template(const T* input, T* output, T min, T max, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T data = input[idx];
        T tmp= data >min ? data : min;
        output[idx] = tmp < max ? tmp : max;
    }
}


// Test function for `clip_kernel_template` with `float`
void test_clip_template_float(const float* h_input, float* h_output, float min, float max, int size) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    clip_kernel_template<<<blocks, threads>>>(d_input, d_output, min, max, size);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    clip_kernel_template<<<blocks, threads>>>(d_input, d_output, min, max, size);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for clip_kernel_template (float): %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int size = 1024; // Number of elements
    const float min = 0.2f; // Minimum clipping value
    const float max = 0.8f; // Maximum clipping value

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output_float = (float*)malloc(size * sizeof(float));
    float* h_output_half = (float*)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 10) / 10.0f; // Values between 0.0 and 0.9
    }

    // Test single-precision kernel
    printf("Testing clip_kernel (float)...\n");
    test_clip_float(h_input, h_output_float, min, max, size);

    // Test half-precision kernel
    printf("Testing clip_kernel_half (__half)...\n");
    test_clip_half(h_input, h_output_half, min, max, size);

    // Test template kernel
    printf("Testing clip_kernel_template (float)...\n");
    test_clip_template_float(h_input, h_output_float, min, max, size);

    // Verify results
    printf("Verifying results...\n");
    for (int i = 0; i < 10; i++) { // Print first 10 results
        printf("Input: %f, Float Output: %f, Half Output: %f\n",
               h_input[i], h_output_float[i], h_output_half[i]);
    }

    // Free host memory
    free(h_input);
    free(h_output_float);
    free(h_output_half);

    return 0;
}