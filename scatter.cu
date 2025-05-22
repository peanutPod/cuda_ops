#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief 在2D数据表上执行散点更新操作(Scatter)
 * 
 * @param output 输出数组，二维表格
 * @param updates 更新值数组，包含要写入的新数据
 * @param indices 索引数组，指定更新位置的行索引
 * @param pitch 输出数组每行的字节数(包括可能的对齐填充)
 * @param rowSize 每行要复制的实际字节数
 * 
 * @note 实现公式: output[indices[i]] = updates[i]
 * @note 每个线程块处理一行数据
 */
__global__ void scatterKernel(
    char* output,        // 输出数组(目标)
    const char* updates, // 更新值数组(源)
    const int* indices,  // 索引数组
    int pitch,           // 输出数组的行间距(字节)
    int rowSize)         // 每行的实际大小(字节)
{
    // 获取当前批次的目标索引
    int idx = indices[blockIdx.x];
    
    // 计算目标位置的内存地址
    char* pDst = (char*)output + idx * pitch;
    
    // 计算源数据的内存地址
    const char* pSrc = updates + blockIdx.x * rowSize;
    
    // 使用内存拷贝将整行数据从源复制到目标
    memcpy(pDst, pSrc, rowSize);
}

/**
 * @brief 将N维索引转换为1维线性索引
 * 
 * @param output 输出的线性索引数组
 * @param transformCoeff 变换系数数组(实际上是各个维度的步长)
 * @param indices N维索引数组
 * @param sliceRank 索引的维度数量
 * 
 * @note 每个线程块处理一个N维索引，将其转换为一个1维索引
 */
__global__ void transformIdxKernel(
    int* output,               // 输出的线性索引
    const int* transformCoeff, // 变换系数(各维度步长)
    const int* indices,        // 输入的N维索引
    int sliceRank)             // 索引的维度
{
    // 计算当前处理的N维索引在indices数组中的起始位置
    const int* idx = indices + sliceRank * blockIdx.x;
    
    // 初始化线性索引
    int transformedIdx = 0;
    
    // 对每个维度进行处理
    for (int i = 0; i < sliceRank; i++)
    {
        // 累加每个维度的贡献: idx[i] * stride[i]
        transformedIdx += idx[i] * transformCoeff[i];
    }
    
    // 存储计算得到的线性索引
    output[blockIdx.x] = transformedIdx;
}

/**
 * @brief 执行N维散点更新操作(ScatterND)
 * 
 * @param stream CUDA流
 * @param transformCoeff 变换系数数组(维度步长)
 * @param nOutputDims 输出张量的维度数
 * @param sliceRank 索引的维度数
 * @param nRows 要处理的行数(批次大小)
 * @param rowSize 每行数据的大小(字节)
 * @param copySize 需要复制的总数据大小
 * @param sizeOfElementInBytes 元素大小(字节)
 * @param index 索引数据指针
 * @param updates 更新值数据指针
 * @param data 初始数据指针
 * @param output 输出数据指针
 * @param workspace 工作空间指针(临时存储)
 */
void scatterNDInference(
    cudaStream_t stream,        // CUDA流
    int* transformCoeff,        // 变换系数(主机内存)
    int nOutputDims,            // 输出维度数
    int sliceRank,              // 切片维度
    int nRows,                  // 行数(批次)
    int rowSize,                // 行大小(字节)
    int copySize,               // 需要拷贝的总大小
    int sizeOfElementInBytes,   // 元素大小
    const void* index,          // 索引数据
    const void* updates,        // 更新数据
    const void* data,           // 初始数据
    void* output,               // 输出数据
    void* workspace)            // 工作空间
{
    // 类型转换，使指针类型更明确
    const int* _index = (const int*)(index);
    const char* _updates = (const char*)(updates);
    char* _output = (char*)(output);
    
    // 划分工作空间
    int* wo = (int*)(workspace);
    // transformedIdx存储在工作空间的后半部分
    int* transformedIdx = wo + sizeof(int)*nOutputDims;
    // deviceTransformCoeff存储在工作空间的前半部分
    int* deviceTransformCoeff = wo;
    
    // 将变换系数从主机内存复制到设备内存
    cudaMemcpy(workspace, transformCoeff, sizeof(int) * nOutputDims, cudaMemcpyHostToDevice);
    
    // 启动内核，将N维索引转换为线性索引
    // 配置：nRows个线程块，每块1个线程
    transformIdxKernel<<<nRows, 1, 0, stream>>>(transformedIdx, deviceTransformCoeff, _index, sliceRank);
    
    // 将初始数据复制到输出缓冲区
    cudaMemcpy(output, data, copySize, cudaMemcpyDeviceToDevice);
    
    // 启动散点更新内核，使用计算出的线性索引更新输出数据
    // 注意：这里假设输出数组的行间距等于行大小乘以4(无填充)
    scatterKernel<<<nRows, 1, 0, stream>>>(_output, _updates, transformedIdx, rowSize * 4, rowSize * 4);
}