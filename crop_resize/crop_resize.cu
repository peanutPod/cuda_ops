#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <type_traits>



__device__ inline __half readUint8FromFp32(const float* image_ptr, int idx, int bytepos) {
    // Reinterpret the float value as an integer

    uint32_t packed_value = reinterpret_cast<const uint32_t*>(image_ptr)[idx];
    // Extract the 8-bit values from the packed 32-bit float
    uint8_t value = (packed_value >> bytepos * 8) & 0xFF;
    return static_cast<__half>(value);
}

// read int8 data from image_ptr and convert to float
__device__ inline __half readInt8(const int8_t* ptr, int idx) { return static_cast<__half>(ptr[idx]); }

template <typename T_OUT>
__device__ inline void process_output(int out_idx, float normalized_value, float quant_scale, T_OUT* crops_ptr,
                                      int nthreads) {}

template <>
__device__ inline void process_output<float>(int out_idx, float normalized_value, float quant_scale, float* crops_ptr,
                                             int nthreads) {
    if (out_idx >= 0 && out_idx < nthreads) {
        crops_ptr[out_idx] = normalized_value;
    }
}

template <>
__device__ inline void process_output<__half>(int out_idx, float normalized_value, float quant_scale, __half* crops_ptr,
                                              int nthreads) {
    if (out_idx >= 0 && out_idx < nthreads) {
        crops_ptr[out_idx] = __float2half(normalized_value);
    }
}

template <>
__device__ inline void process_output<int8_t>(int out_idx, float normalized_value, float quant_scale, int8_t* crops_ptr,
                                              int nthreads) {
    if (out_idx >= 0 && out_idx < nthreads) {
        crops_ptr[out_idx] = static_cast<int8_t>(normalized_value * quant_scale);
    }
}

template <typename T_IN, typename T_OUT>
__global__ void customerCropResizeKernel(const int nthreads, const T_IN* image_ptr, const float* boxes_ptr,
                                         T_OUT* crops_ptr, int num_boxes, int image_height, int image_width,
                                         int crop_height, int crop_width, int depth, float extrapolation_values_0,
                                         float extrapolation_values_1, float extrapolation_values_2, float means_0,
                                         float means_1, float means_2, float stds_0, float stds_1, float stds_2,
                                         float quant_scale, int resize_mode, int pad_mode, int channel_reverse,
                                         int output_dtype, int input_reinterpreted_fp32) {
    for (int out_idx = threadIdx.x + blockIdx.x * blockDim.x; out_idx < nthreads; out_idx += blockDim.x * gridDim.x) {
        int idx = out_idx;
        // calc output width index
        const int w_idx = idx % crop_width;
        idx /= crop_width;
        // calc output height index
        const int h_idx = idx % crop_height;
        idx /= crop_height;
        // calc output channel index
        const int c_idx = idx % depth;
        // calc output box index
        const int n_idx = idx / depth;
        const float box_x1 = boxes_ptr[n_idx * 4];
        const float box_y1 = boxes_ptr[n_idx * 4 + 1];
        const float box_x2 = boxes_ptr[n_idx * 4 + 2];
        const float box_y2 = boxes_ptr[n_idx * 4 + 3];

        if (n_idx < 0 || c_idx < 0 || h_idx < 0 || w_idx < 0) {
            continue;
        }

        // account for channel reversal,we reverse the channel index
        int actual_c_idx = c_idx;
        if (channel_reverse) {
            actual_c_idx = depth - 1 - c_idx;
        }

        // Get mean and std using the actual channel index
        const float mean = (actual_c_idx == 0) ? means_0 : (actual_c_idx == 1) ? means_1 : means_2;
        const float std = (actual_c_idx == 0) ? stds_0 : (actual_c_idx == 1) ? stds_1 : stds_2;
        const float extrapolation_value = (actual_c_idx == 0)
                                              ? extrapolation_values_0
                                              : (actual_c_idx == 1) ? extrapolation_values_1 : extrapolation_values_2;

        // calc box width scale and height scale
        float box_width = (box_x2 - box_x1);
        float box_height = (box_y2 - box_y1);
        const float h_scale = (crop_height > 1 && box_height > 1) ? (crop_height) / (box_height * 1.0f) : 0.f;
        const float w_scale = (crop_width > 1 && box_width > 1) ? (crop_width) / (box_width * 1.0f) : 0.f;
        // keep image original ratio
        float scale = min(h_scale, w_scale);

        int scaled_width = roundf(box_width * scale);
        int scaled_height = roundf(box_height * scale);

        // calculate offset according to pad_mode
        int offset_x = 0;
        int offset_y = 0;
        switch (pad_mode) {
            case 0:
                // top-left
                offset_x = 0;
                offset_y = 0;
                break;
            case 1:
                // bottom-right
                offset_x = crop_width - scaled_width;
                offset_y = crop_height - scaled_height;
                break;
            case 2:
                // center
                offset_x = (crop_width - scaled_width) / 2;
                offset_y = (crop_height - scaled_height) / 2;
                break;
        }

        // check if the pixel is outside the box and fill with extrapolation value
        if (w_idx < offset_x || w_idx >= offset_x + scaled_width || h_idx < offset_y ||
            h_idx >= offset_y + scaled_height) {
            // fill with extrapolation value using the actual channel
            float normalized_value = (extrapolation_value - mean) / std;
            if (out_idx >= 0 && out_idx < nthreads) {
                process_output(out_idx, normalized_value, quant_scale, crops_ptr, nthreads);
            }
            continue;
        }

        __half pixel_value = 0.0f;

        float in_x = box_x1 + (w_idx - offset_x) / scale;
        float in_y = box_y1 + (h_idx - offset_y) / scale;
        if (resize_mode == 0) {
            // nearest interpolation
            // calc nearest x and y
            int nearest_x = static_cast<int>(roundf(in_x));
            int nearest_y = static_cast<int>(roundf(in_y));
            // make sure the point is in the roi area
            nearest_x = min(max(nearest_x, static_cast<int>(box_x1)), static_cast<int>(box_x2));
            nearest_y = min(max(nearest_y, static_cast<int>(box_y1)), static_cast<int>(box_y2));

            int input_idx = nearest_y * image_width * depth + nearest_x * depth + actual_c_idx;
            if (input_idx >= 0 && input_idx < image_height * image_width * depth) {
                pixel_value =
                    input_reinterpreted_fp32
                        ? readUint8FromFp32(reinterpret_cast<const float*>(image_ptr), input_idx / 4, input_idx % 4)
                        : readInt8(reinterpret_cast<const int8_t*>(image_ptr), input_idx);
            } else {
                pixel_value = extrapolation_value;
            }

        } else {
            // biliear interpolation
            int top_y = floor(in_y);
            int bottom_y = ceil(in_y);
            int left_x = floor(in_x);
            int right_x = ceil(in_x);

            float top_weight = in_y - top_y;
            float bottom_weight = 1.0f - top_weight;
            float left_weight = in_x - left_x;
            float right_weight = 1.0f - left_weight;

            top_y = min(max(top_y, static_cast<int>(box_y1)), static_cast<int>(box_y2) - 1);
            bottom_y = min(max(bottom_y, static_cast<int>(box_y1)), static_cast<int>(box_y2) - 1);
            left_x = min(max(left_x, static_cast<int>(box_x1)), static_cast<int>(box_x2) - 1);
            right_x = min(max(right_x, static_cast<int>(box_x1)), static_cast<int>(box_x2) - 1);

            __half top_left = input_reinterpreted_fp32
                                 ? readUint8FromFp32(reinterpret_cast<const float*>(image_ptr),
                                                     (top_y * image_width * depth + left_x * depth + actual_c_idx) / 4,
                                                     (top_y * image_width * depth + left_x * depth + actual_c_idx) % 4)
                                 : readInt8(reinterpret_cast<const int8_t*>(image_ptr),
                                            top_y * image_width * depth + left_x * depth + actual_c_idx);
            __half top_right =
                input_reinterpreted_fp32
                    ? readUint8FromFp32(reinterpret_cast<const float*>(image_ptr),
                                        (top_y * image_width * depth + right_x * depth + actual_c_idx) / 4,
                                        (top_y * image_width * depth + right_x * depth + actual_c_idx) % 4)
                    : readInt8(reinterpret_cast<const int8_t*>(image_ptr),
                               top_y * image_width * depth + right_x * depth + actual_c_idx);
            __half bottom_left =
                input_reinterpreted_fp32
                    ? readUint8FromFp32(reinterpret_cast<const float*>(image_ptr),
                                        (bottom_y * image_width * depth + left_x * depth + actual_c_idx) / 4,
                                        (bottom_y * image_width * depth + left_x * depth + actual_c_idx) % 4)
                    : readInt8(reinterpret_cast<const int8_t*>(image_ptr),
                               bottom_y * image_width * depth + left_x * depth + actual_c_idx);
            __half bottom_right =
                input_reinterpreted_fp32
                    ? readUint8FromFp32(reinterpret_cast<const float*>(image_ptr),
                                        (bottom_y * image_width * depth + right_x * depth + actual_c_idx) / 4,
                                        (bottom_y * image_width * depth + right_x * depth + actual_c_idx) % 4)
                    : readInt8(reinterpret_cast<const int8_t*>(image_ptr),
                               bottom_y * image_width * depth + right_x * depth + actual_c_idx);
            
            __half top = __hmul(top_left, right_weight) + __hmul(top_right, left_weight);
            __half bottom = __hmul(bottom_left, right_weight) + __hmul(bottom_right, left_weight);
            pixel_value = __hmul(top, bottom_weight) + __hmul(bottom, top_weight);
        }

        // Normalize the pixel value
        __half normalized_value = (pixel_value - __float2half(mean)) / __float2half(std);

        if (out_idx >= 0 && out_idx < nthreads) {
            process_output(out_idx, normalized_value, quant_scale, crops_ptr, nthreads);
        }
    }
}


    int output_volume = num_boxes * depth * crop_height * crop_width;
    // set block size and grid size
    int block_size = 512;
    int grid_size = (output_volume + block_size - 1) / block_size;


    customerCropResizeKernel<float, int8_t><<<grid_size, block_size, 4, stream>>>(
                output_volume, static_cast<const float*>(image), static_cast<const float*>(rois),
                static_cast<int8_t*>(output), num_boxes, input_height, input_width, crop_height, crop_width, depth,
                extrapolation_values_0, extrapolation_values_1, extrapolation_values_2, means_0, means_1, means_2,
                stds_0, stds_1, stds_2, quant_scale, resize_mode, pad_mode, channel_reverse, output_dtype,
                input_reinterpreted_fp32);

