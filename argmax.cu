#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat> // For FLT_MAX
#include <chrono> // For CPU timing

// Kernel to find the maximum value and its index
__global__ void argmax_kernel(const float* input, int* max_idx, float* max_val, int size) {
    __shared__ float shared_vals[1024]; // Shared memory for values
    __shared__ int shared_idxs[1024];  // Shared memory for indices

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (idx < size) {
        shared_vals[tid] = input[idx];
        shared_idxs[tid] = idx;
    } else {
        shared_vals[tid] = -FLT_MAX; // Set to minimum float value
        shared_idxs[tid] = -1;
    }
    __syncthreads();

    // Perform block-level reduction to find the maximum value and its index
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && shared_vals[tid] < shared_vals[tid + stride]) {
            shared_vals[tid] = shared_vals[tid + stride];
            shared_idxs[tid] = shared_idxs[tid + stride];
        }
        __syncthreads();
    }

    // Write the block's result to global memory
    if (tid == 0) {
        max_val[blockIdx.x] = shared_vals[0];
        max_idx[blockIdx.x] = shared_idxs[0];
    }
}

// Host function to find the global maximum value and its index
void argmax(const float* h_input, int size, int* h_max_idx, float* h_max_val) {
    const int threads = 1024 *1024;
    const int blocks = (size + threads - 1) / threads;

    // Allocate device memory
    float *d_input, *d_max_val;
    int *d_max_idx;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_max_val, blocks * sizeof(float));
    cudaMalloc(&d_max_idx, blocks * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel to find block-level maxima
    argmax_kernel<<<blocks, threads>>>(d_input, d_max_idx, d_max_val, size);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy block-level results back to host
    float* h_block_max_val = (float*)malloc(blocks * sizeof(float));
    int* h_block_max_idx = (int*)malloc(blocks * sizeof(int));
    cudaMemcpy(h_block_max_val, d_max_val, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_block_max_idx, d_max_idx, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Perform final reduction on the host
    *h_max_val = h_block_max_val[0];
    *h_max_idx = h_block_max_idx[0];
    for (int i = 1; i < blocks; i++) {
        if (h_block_max_val[i] > *h_max_val) {
            *h_max_val = h_block_max_val[i];
            *h_max_idx = h_block_max_idx[i];
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_max_val);
    cudaFree(d_max_idx);

    // Free host memory
    free(h_block_max_val);
    free(h_block_max_idx);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// CPU implementation of argmax
void argmax_cpu(const float* input, int size, int* max_idx, float* max_val) {
    *max_val = input[0];
    *max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (input[i] > *max_val) {
            *max_val = input[i];
            *max_idx = i;
        }
    }
}

int main() {
    const int size = 10240; // Number of elements

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 1000); // Random values between 0 and 999
    }

    // Find the maximum value and its index using GPU
    int h_max_idx_gpu;
    float h_max_val_gpu;
    argmax(h_input, size, &h_max_idx_gpu, &h_max_val_gpu);

    // Find the maximum value and its index using CPU
    int h_max_idx_cpu;
    float h_max_val_cpu;
    argmax_cpu(h_input, size, &h_max_idx_cpu, &h_max_val_cpu);

    // Print the results
    printf("GPU Maximum value: %f at index %d\n", h_max_val_gpu, h_max_idx_gpu);
    printf("CPU Maximum value: %f at index %d\n", h_max_val_cpu, h_max_idx_cpu);

    // Compare results
    if (h_max_idx_gpu == h_max_idx_cpu && fabs(h_max_val_gpu - h_max_val_cpu) < 1e-5) {
        printf("Results match!\n");
    } else {
        printf("Results do not match!\n");
    }

    // Free host memory
    free(h_input);

    return 0;
}