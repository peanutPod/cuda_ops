#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel declaration
__global__ void leaky_relu_kernel_float(const float* input, float* output, const float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float data = input[idx];
        output[idx] = (data > 0) ? data : alpha * data;
    }
}

//和简单的写法无本质区别。单纯的无序计算
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void pReLUKernel(const int n, const float negativeSlope, const float* input, float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = input[i] > 0 ? input[i] : input[i] * negativeSlope;
    }
}

// Test function for single-precision (float)
void test_leaky_relu_float(const float* h_input, float* h_output, float alpha, int size) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Warm-up kernel to initialize GPU and avoid initial overhead
    leaky_relu_kernel_float<<<blocks, threads>>>(d_input, d_output, alpha, size);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);
    
    // Launch kernel with timing
    leaky_relu_kernel_float<<<blocks, threads>>>(d_input, d_output, alpha, size);
    
    // Record the stop event
    cudaEventRecord(stop);
    
    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Test function to compare with template-based implementation
void test_prelu_kernel(const float* h_input, float* h_output, float alpha, int size) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Warm-up kernel
    pReLUKernel<256><<<blocks, threads>>>(size, alpha, d_input, d_output);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);
    
    // Launch template kernel with timing
    pReLUKernel<256><<<blocks, threads>>>(size, alpha, d_input, d_output);
    
    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Template kernel execution time: %f ms\n", milliseconds);

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
    const int size = 10 * 1024 * 1024; // 10M elements for better timing measurement
    const float alpha = 0.1f; // Leaky ReLU alpha value

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output1 = (float*)malloc(size * sizeof(float));
    float* h_output2 = (float*)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 2 == 0) ? -1.0f * i / size : i / size; // Alternating positive and negative values
    }

    // Test standard kernel
    printf("Testing leaky_relu_kernel_float...\n");
    test_leaky_relu_float(h_input, h_output1, alpha, size);

    // Test template kernel for comparison
    printf("\nTesting pReLUKernel (template-based)...\n");
    test_prelu_kernel(h_input, h_output2, alpha, size);

    // Verify results
    printf("\nVerifying results (first 5 elements)...\n");
    for (int i = 0; i < 10; i++) {
        printf("Input: %f, Output1: %f, Output2: %f\n", h_input[i], h_output1[i], h_output2[i]);
    }

    // Check if results match
    bool match = true;
    for (int i = 0; i < size; i++) {
        if (fabs(h_output1[i] - h_output2[i]) > 1e-5) {
            match = false;
            printf("Mismatch at index %d: %f vs %f\n", i, h_output1[i], h_output2[i]);
            break;
        }
    }
    
    if (match) {
        printf("\nBoth implementations produce identical results.\n");
    } else {
        printf("\nResults do not match between implementations.\n");
    }

    // Free host memory
    free(h_input);
    free(h_output1);
    free(h_output2);

    return 0;
}

