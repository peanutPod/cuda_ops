#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cuda_fp16.h>

// L2 normalization across channels but not spatial dimensions, with an optional scaling factor

template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void normalizeNotAcrossSpatialKernel(
        const bool channelShared,    // 是否所有通道共享同一个缩放因子
        const int N,                 // 批次大小
        const int C,                 // 通道数
        const int H,                 // 高度
        const int W,                 // 宽度
        const float eps,             // 用于数值稳定的小常数
        const float* scale,          // 缩放因子
        float* inputData,            // 输入数据
        float* outputData)           // 输出数据
{
    const int dim = C * H * W;                     // 每个批次样本的总元素数
    const int spatialDim = H * W;                  // 每个通道的空间维度大小
    const int tile = 32;                           // 处理的瓦片大小，匹配warp大小
    const int numTile = (spatialDim + tile - 1) / tile;  // 需要的瓦片数量（向上取整)
    
    // 对每个批次的每个瓦片进行处理
    for (int n = blockIdx.x; n < N * numTile; n += gridDim.x)
    {
        // 计算当前批次的输入和输出指针
        float* input = inputData + (n / numTile) * dim;
        float* output = outputData + (n / numTile) * dim;
        
        // 共享内存，用于存储平方和
        __shared__ float sum[tile];
        float localsum = 0.0F;  // 每个线程的局部平方和
        
        // 初始化共享内存
        for (int i = threadIdx.x; i < tile; i += nthds_per_cta)
        {
            sum[i] = 0.0F;
        }
        __syncthreads();  // 确保所有线程完成初始化
        
        // 计算所有通道对应位置元素的平方和
        for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta)
        {
            int row = i / tile;                     // 通道索引
            int col = (n % numTile) * tile + i % tile;  // 空间位置索引
            float data = 0.0F;
            if (col < spatialDim)  // 确保索引不越界
                data = input[row * spatialDim + col];
            localsum += data * data;  // 累加平方值
        }
        
        // 使用原子操作将局部和添加到共享内存
        // threadIdx.x & 31 确保映射到正确的共享内存位置（0-31）
        atomicAdd(&sum[threadIdx.x & 31], localsum);
        __syncthreads();  // 等待所有线程完成平方和计算
        
        // 执行归一化：除以平方和的平方根
        for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta)
        {
            int row = i / tile;
            int col = (n % numTile) * tile + i % tile;
            if (col < spatialDim)
            {
                int offset = row * spatialDim + col;
                // 归一化处理
                output[offset] = input[offset] / sqrt(sum[threadIdx.x & 31] + eps);
            }
        }
        
        // 根据channelShared参数应用缩放因子
        if (channelShared)  // 所有通道共享同一个缩放因子
        {
            for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta)
            {
                int row = i / tile;
                int col = (n % numTile) * tile + i % tile;
                if (col < spatialDim)
                    output[row * spatialDim + col] *= scale[0];  // 使用单一缩放因子
            }
        }
        else  // 每个通道使用不同的缩放因子
        {
            for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta)
            {
                int row = i / tile;
                int col = (n % numTile) * tile + i % tile;
                if (col < spatialDim)
                    output[row * spatialDim + col] *= scale[row];  // 使用通道特定的缩放因子
            }
        }
    }
}


void normalizeNotAcrossSpatialGpu(
    cudaStream_t stream,             // CUDA流
    const bool channelShared,        // 是否通道共享缩放因子
    const int N,                     // 批次大小
    const int C,                     // 通道数
    const int H,                     // 高度
    const int W,                     // 宽度
    const float eps,                 // 数值稳定因子
    const void* scale,               // 缩放因子
    const void* inputData,           // 输入数据
    void* outputData)                // 输出数据
{
    const int BS = 128;              // 每个线程块中的线程数
    const int GS = 256;              // 线程块数量
    
    // 确保线程块大小是32的倍数（warp大小）
    PLUGIN_ASSERT(BS % 32 == 0);
    
    // 启动归一化kernel
    normalizeNotAcrossSpatialKernel<BS><<<GS, BS, 0, stream>>>(
        channelShared, N, C, H, W, eps, (const float*) scale, (float*) inputData, (float*) outputData);
    
    // 检查kernel执行是否有错误
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

__global__ void squareKernel(
    const int n,                     // 元素总数
    const float* x,                  // 输入数据
    float* y)                        // 输出数据（平方结果）
{
    // 对数据进行平方操作
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = x[i] * x[i];  // 计算平方值
    }
}

__global__ void scalChannelKernel(
    const int n,                     // 元素总数
    const int spatialDim,            // 每个通道的空间维度大小
    const float* inputData,          // 输入数据
    const float* scale,              // 每个通道的缩放因子
    float* outputData)               // 输出数据
{
    // 对每个元素应用通道特定的缩放因子
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        // scale[i / spatialDim]找出对应通道的缩放因子
        outputData[i] = inputData[i] / scale[i / spatialDim];
    }
}


void normalizeInference(
    cudaStream_t stream,             // CUDA流
    cublasHandle_t handle,           // cuBLAS句柄
    const bool acrossSpatial,        // 是否跨空间维度归一化
    const bool channelShared,        // 是否通道共享缩放因子
    const int N,                     // 批次大小
    const int C,                     // 通道数
    const int H,                     // 高度
    const int W,                     // 宽度
    const float eps,                 // 数值稳定因子
    const void* scale,               // 缩放因子
    const void* inputData,           // 输入数据
    void* outputData,                // 输出数据
    void* workspace)                 // 临时工作空间
{
    CublasWrapper& mCublasWrapper = getCublasWrapper();  // 获取cuBLAS包装器
    const int dim = C * H * W;                          // 每个样本的元素总数
    
    // 归一化方式1：针对整个特征图（跨通道和空间维度）
    if (acrossSpatial)
    {
        float* input = (float*) const_cast<void*>(inputData);
        float* output = (float*) outputData;
        float* buffer = (float*) workspace;
        
        // 对每个样本独立处理
        for (int n = 0; n < N; ++n)
        {
            // 计算输入的每个元素的平方
            squareKernel<<<(dim + 511) / 512, 512, 0, stream>>>(dim, input, buffer);
            
            float normsqr = 0.0F;
            // 使用cuBLAS计算所有平方值的和
            CUBLAS_CHECK(mCublasWrapper.cublasSasum(handle, dim, buffer, 1, &normsqr));
            
            // 将输入复制到输出
            CUBLAS_CHECK(mCublasWrapper.cublasScopy(handle, dim, input, 1, output, 1));
            
            // 计算平方和的平方根的倒数（用于归一化）
            normsqr = 1 / sqrt(normsqr + eps);
            
            // 使用计算出的归一化因子缩放所有输出
            CUBLAS_CHECK(mCublasWrapper.cublasSscal(handle, dim, &normsqr, output, 1));
            
            // 应用缩放因子
            if (channelShared)  // 所有通道共享同一个缩放因子
            {
                CUBLAS_CHECK(mCublasWrapper.cublasSscal(handle, dim, (float*) scale, output, 1));
            }
            else  // 每个通道使用不同的缩放因子
            {
                // 根据通道应用不同的缩放因子
                scalChannelKernel<<<(dim + 511) / 512, 512, 0, stream>>>(
                    dim, H * W, output, (float*) scale, output);
            }
            
            // 移动指针到下一个样本
            input += dim;
            output += dim;
        }
        return STATUS_SUCCESS;
    }
    // 归一化方式2：仅跨通道归一化，不跨空间维度
    else
    {
        // 调用之前实现的函数
        return normalizeNotAcrossSpatialGpu(
            stream, channelShared, N, C, H, W, eps, scale, inputData, outputData);
    }
}