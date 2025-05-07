#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "./ops_kernel/bboxUtils.h"

/**
 * @brief 计算边界框的面积
 * 
 * @param bbox 输入边界框
 * @return float 返回边界框的面积，如果边界框无效则返回0
 */
template <typename T_BBOX>
__device__ float bboxSize (const Bbox<T_BBOX>& bbox){
    // 检查边界框是否有效（最大坐标应大于最小坐标）
    if (float(bbox.xmax)<float(bbox.xmin) || float(bbox.ymax)<float(bbox.ymin){
        return 0.0f;
    }else{
        // 计算宽度和高度，然后返回面积
        float width = float(bbox.xmax) - float(bbox.xmin);
        float height = float(bbox.ymax) - float(bbox.ymin);
        return width * height;
    }
}

/**
 * @brief 计算两个边界框的交集
 * 
 * @param bbox1 第一个边界框
 * @param bbox2 第二个边界框
 * @param intersect_bbox 输出参数，存储计算得到的交集
 */
template <typename T_BBOX>
__device__ void intersectBbox(const Bbox<T_BBOX>& bbox1, const Bbox<T_BBOX>& bbox2, Bbox<T_BBOX>* intersect_bbox){
    // 检查两个边界框是否有交集
    if (bbox2.x_min > bbox1.x_max || bbox2.x_max < bbox1.x_min ||
        bbox2.y_min > bbox1.y_max || bbox2.y_max < bbox1.y_min) {
        // 如果没有交集，将结果边界框设置为[0,0,0,0]
        intersect_bbox->x_min = T_BBOX(0);
        intersect_bbox->y_min = T_BBOX(0);
        intersect_bbox->x_max = T_BBOX(0);
        intersect_bbox->y_max = T_BBOX(0);
    } else {
        // 计算交集边界框的坐标
        // 交集的左上角是两个边界框左上角的最大值
        intersect_bbox->x_min = max(bbox1.x_min, bbox2.x_min);
        intersect_bbox->y_min = max(bbox1.y_min, bbox2.y_min);
        // 交集的右下角是两个边界框右下角的最小值
        intersect_bbox->x_max = min(bbox1.x_max, bbox2.x_max);
        intersect_bbox->y_max = min(bbox1.y_max, bbox2.y_max);
    }
}

/**
 * @brief half精度浮点数的边界框交集计算的专门实现
 * 
 * @param bbox1 第一个边界框
 * @param bbox2 第二个边界框
 * @param intersect_bbox 输出参数，存储计算得到的交集
 */
template <>
__device__ void intersectBbox<__half>(
    const Bbox<__half>& bbox1,
    const Bbox<__half>& bbox2,
    Bbox<__half>* intersect_bbox)
{
    // 将half类型转换为float进行比较，以提高精度和性能
    if (float(bbox2.xmin) > float(bbox1.xmax)
        || float(bbox2.xmax) < float(bbox1.xmin)
        || float(bbox2.ymin) > float(bbox1.ymax)
        || float(bbox2.ymax) < float(bbox1.ymin))
    {
        // 没有交集时返回[0,0,0,0]
        intersect_bbox->xmin = __half(0);
        intersect_bbox->ymin = __half(0);
        intersect_bbox->xmax = __half(0);
        intersect_bbox->ymax = __half(0);
    }
    else
    {
        // 计算交集边界框，注意使用float进行计算后转回half
        intersect_bbox->xmin = max(float(bbox1.xmin), float(bbox2.xmin));
        intersect_bbox->ymin = max(float(bbox1.ymin), float(bbox2.ymin));
        intersect_bbox->xmax = min(float(bbox1.xmax), float(bbox2.xmax));
        intersect_bbox->ymax = min(float(bbox1.ymax), float(bbox2.ymax));
    }
}

/**
 * @brief 确保边界框坐标正确排序（最小值<=最大值）
 * 
 * @param bbox1 输入边界框
 * @return Bbox<T_BBOX> 返回排序后的边界框
 */
template <typename T_BBOX>
__device__ Bbox<T_BBOX> getDiagonalMinMaxSortedBox(const Bbox<T_BBOX>& bbox1){
    Bbox<T_BBOX> result;
    // 确保xmin <= xmax
    result.x_min = min(bbox1.x_min, bbox1.x_max);
    result.x_max = max(bbox1.x_min, bbox1.x_max);
    // 确保ymin <= ymax
    result.y_min = min(bbox1.y_min, bbox1.y_max);
    result.y_max = max(bbox1.y_min, bbox1.y_max);
    return result;
}

/**
 * @brief half精度的边界框排序专门实现
 */
template <>
__device__ Bbox<__half> getDiagonalMinMaxSortedBox(const Bbox<__half>& bbox1)
{
    Bbox<__half> result;
    // 使用float进行比较以提高精度
    result.xmin = min(float(bbox1.xmin), float(bbox1.xmax));
    result.xmax = max(float(bbox1.xmin), float(bbox1.xmax));
    result.ymin = min(float(bbox1.ymin), float(bbox1.ymax));
    result.ymax = max(float(bbox1.ymin), float(bbox1.ymax));
    return result;
}

/**
 * @brief 计算两个边界框的Jaccard overlap (IoU)
 * 
 * @param bbox1 第一个边界框
 * @param bbox2 第二个边界框
 * @param normalized 边界框坐标是否已归一化
 * @return float IoU值，范围[0,1]
 */
template <typename T_BBOX>
__device__ float jaccardOverlap(const Bbox<T_BBOX>& bbox1, const Bbox<T_BBOX>& bbox2, const bool normalized){
    // 计算两个边界框的交集
    Bbox<T_BBOX> intersect_bbox;
    
    // 首先确保边界框坐标正确排序
    Bbox<T_BBOX> localbbox1=getDiagonalMinMaxSortedBox(bbox1);
    Bbox<T_BBOX> localbbox2=getDiagonalMinMaxSortedBox(bbox2);
    
    // 计算交集
    intersectBbox(localbbox1, localbbox2, &intersect_bbox);
    float intersect_width,intersect_height;
    
    // 计算交集的宽和高
    intersect_width = float(intersect_bbox.x_max) - float(intersect_bbox.x_min);
    intersect_height = float(intersect_bbox.y_max) - float(intersect_bbox.y_min);
    
    // 只有当交集有正面积时才计算IoU
    if (intersect_width >0 && intersect_height >0){
        float intersect_size = intersect_width * intersect_height;
        float bbox1_size = bboxSize(localbbox1);
        float bbox2_size = bboxSize(localbbox2);
        // IoU = 交集面积 / (A面积 + B面积 - 交集面积)
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    }
    // 注意：这里缺少else分支的返回值，应该返回0.0f
    return 0.0f; // 添加这行确保函数始终有返回值
}

/**
 * @brief 初始化一个空的BboxInfo结构
 * 
 * @param bbox_info 要初始化的BboxInfo指针
 */
template <typename T_BBOX>
__device__ void emptyBboxInfo(BboxInfo<T_BBOX>* bbox_info){
    // 置信度设为0
    bbox_info->conf_score=T_BBOX(0);
    // -2表示未分配标签(-1用于share_location为true时的所有标签)
    bbox_info->label =-2;
    // -1表示无效的边界框索引
    bbox_info->bbox_idx =-1;
    // false表示此边界框不保留
    bbox_info->kept =false;   
}

/**
 * @brief 所有类别的非极大值抑制(NMS)核函数
 * 
 * @tparam T_SCORE 分数数据类型
 * @tparam T_BBOX 边界框坐标数据类型
 * @tparam TSIZE 每个线程处理的边界框数量
 * @param num 批处理大小
 * @param num_classes 类别数量
 * @param num_preds_per_class 每个类别的预测框数量
 * @param top_k 每个类别保留的最大检测数
 * @param nms_threshold NMS的IoU阈值
 * @param share_location 是否所有类别共享位置
 * @param isNormalized 边界框坐标是否已归一化
 * @param bbox_data 边界框数据
 * @param beforeNMS_scores NMS前的分数
 * @param beforeNMS_index_array NMS前的索引数组
 * @param afterNMS_scores NMS后的分数
 * @param afterNMS_index_array NMS后的索引数组
 * @param flipXY 是否交换XY坐标
 * @param score_shift 分数偏移量
 */
template <typename T_SCORE, typename T_BBOX, int TSIZE>
__global__ void allClassNMS_kernel(const int num, const int num_classes, const int num_preds_per_class, const int top_k,
    const float nms_threshold, const bool share_location, const bool isNormalized,
    T_BBOX* bbox_data, // bbox_data应为float类型以保留位置信息
    T_SCORE* beforeNMS_scores, int* beforeNMS_index_array, T_SCORE* afterNMS_scores, int* afterNMS_index_array,
    bool flipXY, const float score_shift)
{
    // 共享内存，用于存储哪些边界框应该被保留
    extern __shared__ bool kept_bboxinfo_flag[];
    
    // 对批次中的每个样本进行处理
    for (int i = 0; i < num; i++)
    {
        // 计算当前处理的起始偏移量
        int32_t const offset = i * num_classes * num_preds_per_class + blockIdx.x * num_preds_per_class;
        // 不应写入超过[offset, top_k)范围的数据
        int32_t const max_idx = offset + top_k;
        // 不应读取超过[offset, num_preds_per_class)范围的数据
        int32_t const max_read_idx = offset + min(top_k, num_preds_per_class);
        // 边界框索引的偏移量
        int32_t const bbox_idx_offset = i * num_preds_per_class * (share_location ? 1 : num_classes);
        
        // 线程局部数据
        int loc_bboxIndex[TSIZE];         // 存储边界框索引
        Bbox<T_BBOX> loc_bbox[TSIZE];     // 存储边界框坐标
        
        // 初始化边界框、边界框信息和保留标志
        // 消除共享内存RAW(读后写)冒险
        __syncthreads();
#pragma unroll
        for (int t = 0; t < TSIZE; t++)
        {
            // 计算当前线程处理的索引
            const int cur_idx = threadIdx.x + blockDim.x * t;
            const int item_idx = offset + cur_idx;
            
            // 初始化所有输出数据
            if (item_idx < max_idx)
            {
                // 不访问超出读取边界的数据
                if (item_idx < max_read_idx)
                {
                    loc_bboxIndex[t] = beforeNMS_index_array[item_idx];
                }
                else
                {
                    loc_bboxIndex[t] = -1;
                }
                
                if (loc_bboxIndex[t] != -1)
                {
                    // 计算边界框数据的实际索引
                    const int bbox_data_idx = share_location ? 
                        (loc_bboxIndex[t] % num_preds_per_class + bbox_idx_offset) : loc_bboxIndex[t];
                    
                    // 根据flipXY参数加载边界框坐标
                    loc_bbox[t].xmin = flipXY ? bbox_data[bbox_data_idx * 4 + 1] : bbox_data[bbox_data_idx * 4 + 0];
                    loc_bbox[t].ymin = flipXY ? bbox_data[bbox_data_idx * 4 + 0] : bbox_data[bbox_data_idx * 4 + 1];
                    loc_bbox[t].xmax = flipXY ? bbox_data[bbox_data_idx * 4 + 3] : bbox_data[bbox_data_idx * 4 + 2];
                    loc_bbox[t].ymax = flipXY ? bbox_data[bbox_data_idx * 4 + 2] : bbox_data[bbox_data_idx * 4 + 3];
                    kept_bboxinfo_flag[cur_idx] = true;  // 初始时保留此边界框
                }
                else
                {
                    kept_bboxinfo_flag[cur_idx] = false; // 无效索引，不保留
                }
            }
            else
            {
                kept_bboxinfo_flag[cur_idx] = false;     // 超出范围，不保留
            }
        }
        
        // 过滤掉与高分数边界框重叠较大的低分数边界框
        int ref_item_idx = offset;        // 参考项索引从offset开始
        
        int32_t ref_bbox_idx = -1;        // 参考边界框索引
        if (ref_item_idx < max_read_idx)
        {
            // 计算参考边界框的实际索引
            ref_bbox_idx = share_location
                ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class + bbox_idx_offset)
                : beforeNMS_index_array[ref_item_idx];
        }
        
        // 当参考边界框有效且未超出读取范围时
        while ((ref_bbox_idx != -1) && ref_item_idx < max_read_idx)
        {
            // 加载参考边界框
            Bbox<T_BBOX> ref_bbox;
            ref_bbox.xmin = flipXY ? bbox_data[ref_bbox_idx * 4 + 1] : bbox_data[ref_bbox_idx * 4 + 0];
            ref_bbox.ymin = flipXY ? bbox_data[ref_bbox_idx * 4 + 0] : bbox_data[ref_bbox_idx * 4 + 1];
            ref_bbox.xmax = flipXY ? bbox_data[ref_bbox_idx * 4 + 3] : bbox_data[ref_bbox_idx * 4 + 2];
            ref_bbox.ymax = flipXY ? bbox_data[ref_bbox_idx * 4 + 2] : bbox_data[ref_bbox_idx * 4 + 3];
            
            // 消除共享内存RAW冒险
            __syncthreads();
            
            // 计算当前参考边界框与每个线程负责的边界框的IoU
            for (int t = 0; t < TSIZE; t++)
            {
                const int cur_idx = threadIdx.x + blockDim.x * t;
                const int item_idx = offset + cur_idx;
                
                // 只检查标记为保留的边界框，并且只与索引大于参考索引的边界框比较
                // 这样可以确保高分数的边界框先处理，低分数的边界框可能被抑制
                if ((kept_bboxinfo_flag[cur_idx]) && (item_idx > ref_item_idx))
                {
                    // 如果IoU大于阈值，则不保留当前边界框
                    if (jaccardOverlap(ref_bbox, loc_bbox[t], isNormalized) > nms_threshold)
                    {
                        kept_bboxinfo_flag[cur_idx] = false;
                    }
                }
            }
            __syncthreads();
            
            // 寻找下一个有效的参考边界框（未被抑制的）
            do
            {
                ref_item_idx++;
            } while (ref_item_idx < max_read_idx && !kept_bboxinfo_flag[ref_item_idx - offset]);
            
            // 移动到下一个有效点
            if (ref_item_idx < max_read_idx)
            {
                // 更新参考边界框索引
                ref_bbox_idx = share_location
                    ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class + bbox_idx_offset)
                    : beforeNMS_index_array[ref_item_idx];
            }
        }
        
        // 存储NMS后的结果数据
        for (int t = 0; t < TSIZE; t++)
        {
            const int cur_idx = threadIdx.x + blockDim.x * t;
            const int read_item_idx = offset + cur_idx;
            const int write_item_idx = (i * num_classes * top_k + blockIdx.x * top_k) + cur_idx;
            
            /*
             * 如果不保留边界框：
             * 设置分数为score_shift
             * 设置边界框索引为-1
             */
            if (read_item_idx < max_idx)
            {
                // 保留的边界框使用原始分数，否则使用score_shift
                afterNMS_scores[write_item_idx] = kept_bboxinfo_flag[cur_idx] ? 
                    T_SCORE(beforeNMS_scores[read_item_idx]) : T_SCORE(score_shift);
                // 保留的边界框使用原始索引，否则使用-1
                afterNMS_index_array[write_item_idx] = kept_bboxinfo_flag[cur_idx] ? loc_bboxIndex[t] : -1;
            }
        }
    }
}