#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>

// Kernel declaration
__global__ void sum_square_kernel(const float* input, float* output, int size);

// Kernel definition
__global__ void sum_square_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate the sum of squares
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += input[i] * input[i]; 
        }
        output[idx] = sum;
    }
}

// Test function for `sum_square_kernel`
void test_sum_square_kernel(const float* h_input, float* h_output, int size) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    int threads = 512;
    int blocks = (size + threads - 1) / threads;
    sum_square_kernel<<<blocks, threads>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for sum_square_kernel: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void sum_square_kernel_opt(const float* input,float* output,int size){
    __shared__ float shared_data[1024];
    int tid = threadIdx.x;
    int idx= blockIdx.x * blockDim.x + tid;

    shared_data[tid]=(idx < size) ? input[idx] * input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x /2;stride >0; stride /= 2){
        if (tid < stride){
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();

    }
    if (tid == 0){
        output[blockIdx.x] = shared_data[0];
    }
}


__global__ void sum_square_kernel_warp(const float* input, float* partial_sums, int size) {
    __shared__ float shared_data[1024]; // Shared memory for partial sums
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    shared_data[tid] = (idx < size) ? input[idx] * input[idx] : 0.0f;
    __syncthreads();

    // Perform block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Perform warp-level reduction using warp shuffle
    if (tid < 32) {
        float val = shared_data[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        shared_data[tid] = val;
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}


__global__ void sum_square_kernel_warp_optimized(const float* input, float* partial_sums, int size) {
    __shared__ float shared_data[1024]; // Shared memory for partial sums
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    float val = (idx < size) ? input[idx] * input[idx] : 0.0f;
    shared_data[tid] = val;
    __syncthreads();

    // Perform block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Perform warp-level reduction directly in registers
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        partial_sums[blockIdx.x] = val;
    }
}

void compute_sum_square(const float* h_input, float* h_output, int size) {
    float *d_input, *d_partial_sums;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    sum_square_kernel_warp<<<blocks, threads>>>(d_input, d_partial_sums, size);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);



    // Launch the optimized kernel
    sum_square_kernel_warp_optimized<<<blocks, threads>>>(d_input, d_partial_sums, size);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for sum_square_kernel: %f ms\n", milliseconds);

    // Copy partial sums back to host
    float* h_partial_sums = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute the final sum on the host
    float total_sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total_sum += h_partial_sums[i];
    }

    // Store the result in the output
    *h_output = total_sum;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_partial_sums);

    // Free host memory
    free(h_partial_sums);
}

int main() {
    const int size = 1024*1024; // Number of elements

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output = (float*)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 10) + 1.0f; // Values between 1.0 and 10.0
    }

    // Test sum_square_kernel
    printf("Testing sum_square_kernel...\n");
    test_sum_square_kernel(h_input, h_output, size);


    printf("Testing optimized sum_square_kernel...\n");
    compute_sum_square(h_input, h_output, size);

    // Verify results
    printf("Verifying results...\n");
    float expected_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        expected_sum += h_input[i] * h_input[i];
    }

    for (int i = 0; i < 10; i++) { // Print first 10 results
        printf("Output[%d]: %f, Expected: %f\n", i, h_output[i], expected_sum);
    }

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}