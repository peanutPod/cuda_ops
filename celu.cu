#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

//celu func: max(0,x) + min(0,alpha*(exp(x/alpha)-1))


__global__ void celu_kernel(const float* input,float* output,const float alpha,int size){
    int idx = blockIdx.x *blockDim.x +threadIdx.x;
    if(idx<size){
        float data=input[idx];
        // output[idx]= fmaxf(0,data) + fminf(0,alpha * expf(data/alpha)-1.0f);
        output[idx]= fmaxf(0,data) + fminf(0,alpha * expf(data/alpha)-1.0f);
    }
}

// CELU function for __half
__global__ void celu_kernel_half(const __half* input, __half* output, const __half alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __half data = input[idx];
#if __CUDA_ARCH__ >= 530
        // Use native half-precision operations if supported
        output[idx] = __hmax(__float2half(0), data);
        output[idx] = __hadd(output[idx], __hmin(__float2half(0), __hsub(__hmul(alpha, hexp(__hdiv(data, alpha))), __float2half(1.0f))));
#else
        // Fallback to single-precision arithmetic
        float data_f = __half2float(data);
        float alpha_f = __half2float(alpha);
        float result_f = fmaxf(0.0f, data_f) + fminf(0.0f, alpha_f * (expf(data_f / alpha_f) - 1.0f));
        output[idx] = __float2half(result_f);
#endif
    }
}




// Host function to launch CELU kernel
void celu(const float *h_input, float *h_output, float alpha, int size) {
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 512; // Optimal thread count per block
    int blocks = (size + threads - 1) / threads; // Calculate number of blocks
    //warm-up kernel launch
    celu_kernel<<<blocks, threads>>>(d_input, d_output, alpha, size);

    cudaEvent_t start ,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    celu_kernel<<<blocks, threads>>>(d_input, d_output, alpha, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for CELU kernel: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


void test_celu_half(const float *h_input_float, float *h_output_float, float alpha, int size) {
    // Allocate host memory for half-precision
    __half *h_input_half = (__half *)malloc(size * sizeof(__half));
    __half *h_output_half = (__half *)malloc(size * sizeof(__half));

    // Convert input data from float to __half
    for (int i = 0; i < size; i++) {
        h_input_half[i] = __float2half(h_input_float[i]);
    }

    // Allocate device memory
    __half *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(__half));
    cudaMalloc(&d_output, size * sizeof(__half));

    // Copy input data to device
    cudaMemcpy(d_input, h_input_half, size * sizeof(__half), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 512; // Optimal thread count per block
    int blocks = (size + threads - 1) / threads; // Calculate number of blocks

    // Warm-up kernel launch
    celu_kernel_half<<<blocks, threads>>>(d_input, d_output, __float2half(alpha), size);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure execution time
    cudaEventRecord(start);
    celu_kernel_half<<<blocks, threads>>>(d_input, d_output, __float2half(alpha), size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for CELU kernel (half-precision): %f ms\n", milliseconds);

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


int main() {
    const int size = 1 << 20; // 1M elements
    const float alpha = 1.0f;

    // Allocate host memory
    float *h_input = (float *)malloc(size * sizeof(float));
    float *h_output = (float *)malloc(size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 2 == 0) ? -1.0f * i / size : i / size;
    }

    // Run CELU
    celu(h_input, h_output, alpha, size);
    test_celu_half(h_input, h_output, alpha, size);


    // Cleanup
    free(h_input);
    free(h_output);

    return 0;
}