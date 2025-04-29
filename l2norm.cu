#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include<math.h>

__global__ void L2Norm(const float* input, float* output, const int size, const float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate the L2 norm
        float norm = 0.0f;
        for (int i = 0; i < size; i++) {
            norm += input[i] * input[i]; 
        }
        norm = sqrt(norm);
        
        // Normalize the input and apply the scale factor
        output[idx] = (input[idx] / norm) * scale;
    }
}

// Test function for `L2Norm`
void test_L2Norm(const float* h_input, float* h_output, int size, float scale) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;
    L2Norm<<<blocks, threads>>>(d_input, d_output, size, scale);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    L2Norm<<<blocks, threads>>>(d_input, d_output, size, scale);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for L2Norm kernel: %f ms\n", milliseconds);

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
    const int size = 1024 * 1024; // Number of elements
    const float scale = 2.0f; // Scale factor

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output = (float*)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 10) + 1.0f; // Values between 1.0 and 10.0
    }

    // Test optimized L2Norm
    printf("Testing optimized L2Norm...\n");
    test_L2Norm(h_input, h_output, size, scale);

    // Verify results
    printf("Verifying results...\n");

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}