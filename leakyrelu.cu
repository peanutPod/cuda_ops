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

// Test function for single-precision (float)
void test_leaky_relu_float(const float* h_input, float* h_output, float alpha, int size) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    leaky_relu_kernel_float<<<blocks, threads>>>(d_input, d_output, alpha, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int size = 1024; // Number of elements
    const float alpha = 0.1f; // Leaky ReLU alpha value

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output = (float*)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 2 == 0) ? -1.0f * i / size : i / size; // Alternating positive and negative values
    }

    // Test single-precision kernel
    printf("Testing leaky_relu_kernel_float (float)...\n");
    test_leaky_relu_float(h_input, h_output, alpha, size);

    // Verify results
    printf("Verifying results...\n");
    for (int i = 0; i < 10; i++) { // Print first 10 results
        printf("Input: %f, Output: %f\n", h_input[i], h_output[i]);
    }

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}

