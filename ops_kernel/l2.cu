#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute L2 norm in a single kernel
__global__ void compute_L2Norm_kernel(const float* input, double* result, int size) {
    __shared__ double shared_data[1024]; // Shared memory for partial sums
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Load data into shared memory and compute square
    double val = (idx < size) ? (double)input[idx] * (double)input[idx] : 0.0;
    shared_data[tid] = val;
    __syncthreads();

    // Step 2: Perform block-level reduction using warp shuffle
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: Perform warp-level reduction
    if (tid < 32) {
        double warp_val = shared_data[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, offset);
        }
        shared_data[tid] = warp_val;
    }
    __syncthreads();

    // Step 4: Write the block's partial sum to global memory
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

double compute_L2Norm(const float* h_input, int size) {
    float *d_input;
    double *d_result;
    double h_result = 0.0;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_result, sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize result on device
    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

    // Preheat GPU with a dummy kernel
    compute_L2Norm_kernel<<<blocks, threads>>>(d_input, d_result, size);
    cudaDeviceSynchronize();

    // Reset result on device
    h_result = 0.0;
    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernel
    compute_L2Norm_kernel<<<blocks, threads>>>(d_input, d_result, size);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy the final result back to the host
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_result);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Return the result
    return sqrt(h_result);
}

// High-precision CPU implementation of L2 norm
double compute_L2Norm_CPU(const float* input, int size) {
    double sum_of_squares = 0.0;
    for (int i = 0; i < size; i++) {
        sum_of_squares += (double)input[i] * (double)input[i];
    }
    return sqrt(sum_of_squares);
}

int main() {
    const int size = 1024 * 1024; // Number of elements

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 10) + 1.0f; // Values between 1.0 and 10.0
    }

    // Compute L2 norm on GPU
    printf("Computing L2 norm on GPU...\n");
    double l2_norm_gpu = compute_L2Norm(h_input, size);

    // Compute L2 norm on CPU with high precision
    printf("Computing L2 norm on CPU (high precision)...\n");
    double l2_norm_cpu = compute_L2Norm_CPU(h_input, size);

    // Print the results
    printf("L2 Norm (GPU): %.15f\n", l2_norm_gpu);
    printf("L2 Norm (CPU, high precision): %.15f\n", l2_norm_cpu);

    // Free host memory
    free(h_input);

    return 0;
}