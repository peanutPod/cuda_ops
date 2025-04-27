#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel declarations
__global__ void gelu_kernel_float(const float* input, float* output, const int size, const int approximate);

//functiong description: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gelu


__global__ void gelu_kernel_float(const float* input, float* output ,const int size,const int approximate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float result;
        if (approximate == 0) {
            result = x * 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        } else {
            result = x * 0.5f * (1.0f + erfcf(x/1.4142135623730951f));
        }
        output[idx] = result;
    }
}

// Test function for single-precision (float)
void test_gelu_float(const float* h_input, float* h_output, int size, int approximate) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_kernel_float<<<blocks, threads>>>(d_input, d_output, size, approximate);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


int main() {
    const int size = 1024; // Number of elements
    const int approximate = 0; // 0 for approximate GELU, 1 for exact GELU

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output_float = (float*)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 10) / 10.0f; // Values between 0.0 and 0.9
    }

    // Test single-precision kernel
    printf("Testing gelu_kernel_float (float)...\n");
    test_gelu_float(h_input, h_output_float, size, approximate);


    // Verify results
    printf("Verifying results...\n");
    for (int i = 0; i < 10; i++) { // Print first 10 results
        printf("Input: %f, Float Output: %f\n",
               h_input[i], h_output_float[i]);
    }

    // Free host memory
    free(h_input);
    free(h_output_float);

    return 0;
}


