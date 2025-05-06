#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

__global__ void add_kernel(
    const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

//使用__restrict__关键字来告诉编译器指针不重叠,对于简单的计算逻辑，编译器已经充分优化，无优化空间
__global__ void add_kernel_optimized(
    const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

//使用共享内存的方式并不能对add算子进行优化
__global__ void add_kernel_shared(
    const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int size) {
    __shared__ float shared_a[256];
    __shared__ float shared_b[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        shared_a[tid] = a[idx];
        shared_b[tid] = b[idx];
    }
    __syncthreads();

    // Perform addition
    if (idx < size) {
        c[idx] = shared_a[tid] + shared_b[tid];
    }
}

void test_add_kernel() {
    const int size = 1024 * 1024; // Number of elements
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(size - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy input data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Warmup kernel (preheat GPU)
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < size; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            printf("Mismatch at index %d: GPU result %f, expected %f\n", i, h_c[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("Running test for add_kernel...\n");
    test_add_kernel();
    return 0;
}

