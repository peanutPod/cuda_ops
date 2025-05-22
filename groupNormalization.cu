#include <cuda_runtime.h>  // CUDA运行时库
#include <cuda_fp16.h>     // 半精度浮点数(FP16)支持

/**
 * @brief 对归一化后的特征图进行缩放和平移操作(inplace操作)
 * 
 * @tparam T 数据类型(float或__half)
 * @tparam TPB 每个线程块的线程数(Thread Per Block)
 * @param inOut 输入/输出数据指针(指向同一块内存)
 * @param ld 每个通道的元素数量(通常是H*W)
 * @param beta 平移参数(偏置)
 * @param gamma 缩放参数(尺度)
 * 
 * @note 网格维度为(blocks_per_col, C, B)，表示(每列的块数, 通道数, 批次大小)
 * @note 该函数实现公式: y = gamma * x + beta
 */
template <typename T, unsigned TPB>
__global__ void scaleShiftChannelsInplaceKernel(T* inOut, const int ld, const float* beta, const float* gamma)
{
    // 网格结构解释:
    // blockIdx.z = 批次索引(batch index)
    // blockIdx.y = 通道索引(channel index)
    // blockIdx.x = 每列的块索引(block per column)
    
    // 获取当前通道对应的beta和gamma值
    const T b = beta[blockIdx.y];
    const T g = gamma[blockIdx.y];

    // 计算当前处理的数据在全局内存中的偏移量
    // offset = (batch_idx * num_channels + channel_idx) * elements_per_channel
    const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * ld;

    // 计算当前线程处理的元素索引
    const int tx = blockIdx.x * TPB + threadIdx.x;

    // 确保线程不会处理超出范围的元素
    if (tx < ld)
    {
        // 应用缩放和平移: output = gamma * input + beta
        inOut[offset + tx] = g * inOut[offset + tx] + b;
    }
}

/*
 * 内核启动配置(已注释)
 * 
 * constexpr int TPB = 256;  // 每个块256个线程，GPU warp大小(32)的倍数
 * const int colBlocks = (channelVolume + TPB - 1) / TPB;  // 计算每列需要的块数
 * const dim3 grid(colBlocks, C, B);  // 创建3D网格: (每列块数, 通道数, 批次大小)
 * 
 * // 启动内核
 * scaleShiftChannelsInplaceKernel<T, TPB><<<grid, TPB, 0, stream>>>(inOut, channelVolume, beta, gamma);
 */