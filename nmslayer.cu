#include <cuda_runtime.h>  // CUDA运行时API
#include <cuda_fp16.h>     // 半精度浮点数支持
#include "./ops_kernel/bboxUtils.h"  // 边界框工具函数

/**
 * @brief 计算两个边界框的IoU(交并比)
 * 
 * @tparam TFloat 边界框坐标的数据类型
 * @param a 第一个边界框
 * @param b 第二个边界框
 * @return float IoU值，范围[0,1]
 * 
 * @note 可在设备和主机上执行(__device__ __host__)
 * @note 计算包含边框像素(+1)的面积计算方式
 */
template <typename TFloat>
__device__ __host__ inline float IoU(const Bbox<TFloat>&a, const Bbox<TFloat>&b) {
    // 计算两个框的交集区域坐标
    TFloat left = max(a.xmin, b.xmin), right = min(a.xmax, b.xmax);
    TFloat top = max(a.ymin, b.ymin), bottom = min(a.ymax, b.ymax);
    
    // 计算交集区域的宽和高(加1表示包含边界像素)
    TFloat width = max((TFloat)(right - left + (TFloat) 1.0), (TFloat) 0.0);
    TFloat height = max((TFloat)(bottom - top + (TFloat) 1.0), (TFloat) 0.0);
    
    // 计算交集面积
    TFloat interS = width * height;
    
    // 计算两个边界框的面积
    TFloat Sa = (a.xmax - a.xmin + (TFloat) 1) * (a.ymax - a.ymin + (TFloat) 1);
    TFloat Sb = (b.xmax - b.xmin + (TFloat) 1) * (b.ymax - b.ymin + (TFloat) 1);
    
    // 返回IoU = 交集面积 / (A面积 + B面积 - 交集面积)
    return (float) interS / (float) (Sa + Sb - interS);
}

/**
 * @brief 小批量数据的NMS kernel实现
 * 
 * @tparam T_PROPOSALS 输入边界框数据类型
 * @tparam T_ROIS 输出边界框数据类型
 * @param DIM 线程块大小
 * @param TSIZE 每个线程处理的边界框数量
 * @param propSize 每个批次的提议数量
 * @param preNmsProposals NMS前的边界框
 * @param afterNmsProposals NMS后的边界框
 * @param preNmsTopN NMS前保留的最大边界框数量
 * @param nmsThres NMS阈值(IoU阈值)
 * @param afterNmsTopN NMS后保留的最大边界框数量
 * 
 * @note 使用共享内存跟踪保留的边界框
 * @note __launch_bounds__(DIM)提供线程块大小信息供编译器优化
 */
template <typename T_PROPOSALS, typename T_ROIS, int DIM, int TSIZE>
__global__ __launch_bounds__(DIM) void nmsKernel1(const int propSize,
                                                  Bbox<T_PROPOSALS> const* __restrict__ preNmsProposals,
                                                  T_ROIS* __restrict__ afterNmsProposals,
                                                  const int preNmsTopN,
                                                  const float nmsThres,
                                                  const int afterNmsTopN)
{
    // 共享内存数组，存储边界框是否保留的标志
    __shared__ bool kept_boxes[TSIZE * DIM];
    
    // 计数已保留的边界框数量
    int kept = 0;
    
    // 当前批次在输入数组中的偏移
    int batch_offset = blockIdx.x * propSize;
    
    // 当前批次可处理的最大边界框索引
    int max_box_idx = batch_offset + preNmsTopN;
    
    // 当前批次在输出数组中的偏移
    int batch_offset_out = blockIdx.x * afterNmsTopN;

    // 线程本地数组，存储边界框信息
    int flag_idx[TSIZE];           // kept_boxes数组的索引
    int boxes_idx[TSIZE];          // 输入边界框的索引
    Bbox<T_PROPOSALS> cur_boxes[TSIZE]; // 当前线程处理的边界框

    // 初始化kept_boxes数组
#pragma unroll // 指示编译器展开循环，提高性能
    for (int i = 0; i < TSIZE; i++)
    {
        // 计算当前线程处理的边界框索引
        boxes_idx[i] = threadIdx.x + batch_offset + DIM * i;
        flag_idx[i] = threadIdx.x + DIM * i;

        if (boxes_idx[i] < max_box_idx)
        {
            // 加载边界框数据到线程本地变量
            cur_boxes[i] = preNmsProposals[boxes_idx[i]];
            kept_boxes[flag_idx[i]] = true; // 初始标记为保留
        }
        else
        {
            // 超出有效边界框范围，标记为不保留
            kept_boxes[flag_idx[i]] = false;
            boxes_idx[i] = -1.0f; // 无效索引
            flag_idx[i] = -1.0f;
        }
    }

    // 从第一个边界框开始处理
    int ref_box_idx = 0 + batch_offset;

    // 移除重叠的边界框
    while ((kept < afterNmsTopN) && (ref_box_idx < max_box_idx))
    {
        // 加载参考边界框
        Bbox<T_PROPOSALS> ref_box;
        ref_box = preNmsProposals[ref_box_idx];

#pragma unroll
        for (int i = 0; i < TSIZE; i++)
        {
            // 只处理索引大于参考框的边界框(避免重复比较)
            if (boxes_idx[i] > ref_box_idx)
            {
                // 计算IoU，如果超过阈值则抑制该边界框
                if (IoU(ref_box, cur_boxes[i]) > nmsThres)
                {
                    kept_boxes[flag_idx[i]] = false;
                }
            }
            // 当前框就是参考框，保存到输出数组
            else if (boxes_idx[i] == ref_box_idx)
            {
                afterNmsProposals[(batch_offset_out + kept) * 4 + 0] = ref_box.xmin;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 1] = ref_box.ymin;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 2] = ref_box.xmax;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 3] = ref_box.ymax;
            }
        }
        // 同步所有线程，确保共享内存一致
        __syncthreads();

        // 寻找下一个未被抑制的边界框
        do
        {
            ref_box_idx++;
        } while (!kept_boxes[ref_box_idx - batch_offset] && ref_box_idx < max_box_idx);

        // 已保留边界框数量加1
        kept++;
    }
}

/**
 * @brief 大批量数据的NMS kernel实现
 * 
 * @tparam T_PROPOSALS 输入边界框数据类型
 * @tparam T_ROIS 输出边界框数据类型
 * @param DIM 线程块大小
 * @param TSIZE 每个线程处理的边界框数量
 * @param propSize 每个批次的提议数量
 * @param proposals NMS前的边界框
 * @param filtered NMS后的边界框
 * @param preNmsTopN NMS前保留的最大边界框数量
 * @param nmsThres NMS阈值(IoU阈值)
 * @param afterNmsTopN NMS后保留的最大边界框数量
 * 
 * @note 使用位操作(uint64_t del)跟踪删除的边界框，比共享内存更高效
 */
template <typename T_PROPOSALS, typename T_ROIS, int DIM, int TSIZE>
__global__ __launch_bounds__(DIM) void nmsKernel2(const int propSize,
                                                  Bbox<T_PROPOSALS> const* __restrict__ proposals,
                                                  T_ROIS* __restrict__ filtered,
                                                  const int preNmsTopN,
                                                  const float nmsThres,
                                                  const int afterNmsTopN)
{
    // 指向当前批次提议的指针
    Bbox<T_PROPOSALS> const* cProposals = proposals + blockIdx.x * propSize;

    // 线程本地数组，存储边界框
    Bbox<T_PROPOSALS> t[TSIZE];
    
    // 使用位图记录被删除的边界框，每个位对应一个边界框
    uint64_t del = 0;

    // 加载边界框数据到线程本地变量
    for (int i = 0; i < TSIZE; i++)
    {
        // 确保不超出有效边界框范围
        if (i < TSIZE - 1 || i * DIM + threadIdx.x < preNmsTopN)
        {
            t[i] = cProposals[i * DIM + threadIdx.x];
        }
    }

    // 共享变量，用于线程间通信
    __shared__ Bbox<T_PROPOSALS> last;   // 最近处理的边界框
    __shared__ bool kept;                // 边界框是否保留
    __shared__ int foundBatch;           // 已找到的边界框数量
    
    // 初始化计数器
    if (threadIdx.x == 0)
        foundBatch = 0;

    // 双重循环处理所有边界框
    for (int i = 0; i < TSIZE; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            int offset = i * DIM;
            int index = offset + j;
            // 超出预处理边界框数量则退出
            if (index >= preNmsTopN)
                break;

            // 同步所有线程
            __syncthreads();

            // 只有一个线程执行此段代码
            if (threadIdx.x == j)
            {
                // 检查当前边界框是否被删除
                kept = 0 == (del & ((uint64_t) 1 << i));
                last = t[i];

                // 如果保留，写入输出数组
                if (kept)
                {
                    int cnt = blockIdx.x * afterNmsTopN + foundBatch;
                    filtered[cnt * 4 + 0] = t[i].xmin;
                    filtered[cnt * 4 + 1] = t[i].ymin;
                    filtered[cnt * 4 + 2] = t[i].xmax;
                    filtered[cnt * 4 + 3] = t[i].ymax;
                    foundBatch++;
                }
            }

            // 同步所有线程
            __syncthreads();

            // 如果已达到目标数量，提前返回
            if (foundBatch == afterNmsTopN)
            {
                return;
            }

            // 如果当前边界框被保留，检查它与其他边界框的重叠
            if (kept)
            {
                Bbox<T_PROPOSALS> test = last;

                for (int k = 0; k < TSIZE; k++)
                {
                    // 只处理索引大于当前边界框的边界框(避免重复比较)
                    if (index < k * DIM + threadIdx.x
                        && IoU<T_PROPOSALS>(test, t[k]) > nmsThres)
                    {
                        // 标记重叠边界框为删除
                        del |= (uint64_t) 1 << k;
                    }
                }
            }
        }
    }
}

/* 
// 下面是注释掉的内核选择代码，提供了基于输入大小动态选择内核的机制

// 设置线程块大小为1024
// const int blockSize = 1024;

// 定义宏简化内核函数指针声明
// #define P1(tsize) nmsKernel1<T_PROPOSALS, T_ROIS, blockSize, (tsize)>
// #define P2(tsize) nmsKernel2<T_PROPOSALS, T_ROIS, blockSize, (tsize)>

// 函数指针数组，存储64种不同配置的内核函数
// void (*kernel[64])(int, Bbox<T_PROPOSALS> const*, T_ROIS*, int, float, int) = {
//     P1(1), P1(2), ..., P1(12), P2(13), ..., P2(64)
// };

// 确保输入大小不超过处理能力
// ASSERT_PARAM(preNmsTopN < 64 * blockSize);

// 初始化输出内存为0
// CSC(cudaMemsetAsync(filtered, 0, batch * afterNmsTopN * 4 * sizeof(T_ROIS), stream), STATUS_FAILURE);

// 根据输入大小选择最佳内核和TSIZE参数
// kernel[(preNmsTopN + blockSize - 1) / blockSize - 1]<<<batch, blockSize, 0, stream>>>(propSize,
//                                                                                       (Bbox<T_PROPOSALS>*) proposals,
//                                                                                       (T_ROIS*) filtered,
//                                                                                       preNmsTopN,
//                                                                                       nmsThres,
//                                                                                       afterNmsTopN);
*/