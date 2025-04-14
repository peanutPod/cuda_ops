#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>

const char *ptxCode = R"(
.version 8.5
.target sm_52
.address_size 64

	// .globl	_Z3addPi

.visible .entry _Z3addPi(
	.param .u64 _Z3addPi_param_0
)
{
	.reg .b32 	%r<7>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [_Z3addPi_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	mul.wide.s32 	%rd3, %r4, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.u32 	%r5, [%rd4];
	shl.b32 	%r6, %r5, 1;
	st.global.u32 	[%rd4], %r6;
	ret;

}
)";

int main() {
    // Initialize the CUDA Driver API
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to initialize CUDA Driver API" << std::endl;
        return -1;
    }

    const int size = 5; // Array size
    int h_data[size] = {1, 2, 3, 4, 5}; // Initialize input array
    int *d_data;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_data, size * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy input data from host to device
    err = cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy (Host to Device) failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Load PTX code
    CUmodule module;
    CUfunction kernel;
    CUresult result = cuModuleLoadData(&module, ptxCode);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        std::cerr << "Error loading PTX: " << errorStr << std::endl;
        return -1;
    }
    if (cuModuleGetFunction(&kernel, module, "_Z3addPi") != CUDA_SUCCESS) {
        std::cerr << "Failed to get kernel function" << std::endl;
        return -1;
    }

    // Set kernel parameters
    void *args[] = { &d_data };
    dim3 grid(1);
    dim3 block(size); // One thread per element

    // Launch kernel
    CUresult launchResult = cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, 0, args, 0);
    if (launchResult != CUDA_SUCCESS) {
        std::cerr << "Kernel launch failed" << std::endl;
        return -1;
    }

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy (Device to Host) failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Print result
    std::cout << "Data: ";
    for (int i = 0; i < size; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_data);
    cuModuleUnload(module);
    return 0;
}