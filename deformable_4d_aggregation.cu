#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>


__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


__global__ void deformable_aggregation_kernel(
    const int num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    float *output_ptr = output + idx;
    const int channel_index = idx % num_embeds;
    const int groups_index = channel_index / (num_embeds / num_groups);
    idx /= num_embeds;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int batch_index = idx;

    const int value_cam_stride = num_feat * num_embeds;
    const int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams << 1;
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset = (
        (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride
        + groups_index
    );

    float result = 0;
    for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 2) {
        const float loc_w = sample_location[loc_offset];
        const float loc_h = sample_location[loc_offset + 1];
        
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
                const int scale_offset = scale_start_index[scale_index] * num_embeds;

                const int spatial_shape_ptr = scale_index << 1;
                const int h = spatial_shape[spatial_shape_ptr];
                const int w = spatial_shape[spatial_shape_ptr + 1];

                const float h_im = loc_h * h - 0.5;
                const float w_im = loc_w * w - 0.5;

                const int value_ptr = value_offset + scale_offset + value_cam_stride * cam_index;
                const float *weights_ptr = (
                    weights + weight_offset + scale_index * num_groups
                    + weight_cam_stride * cam_index
                );
                result += bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_ptr) * *weights_ptr;
            }
        }
    }
    *output_ptr = result;
}


void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups
) {
    const int num_kernels = batch_size * num_pts * num_embeds;
    deformable_aggregation_kernel
        <<<(int)ceil(((double)num_kernels/512)), 512>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts, num_groups
    );
}


void test_deformable_aggregation() {
    // Define test parameters
    const int batch_size = 1;
    const int num_cams = 5;
    const int num_feat = 66;
    const int num_embeds = 33;
    const int num_scale = 2;
    const int num_pts = 256;
    const int num_groups = 2;

    const int spatial_shape_size = num_scale * 2;
    const int scale_start_index_size = num_scale;
    const int sample_location_size = batch_size * num_pts * num_cams * 2;
    const int weights_size = batch_size * num_pts * num_cams * num_scale * num_groups;
    const int mc_ms_feat_size = batch_size * num_cams * num_feat * num_embeds;
    const int output_size = batch_size * num_pts * num_embeds;

    // Allocate host memory
    float* h_mc_ms_feat = (float*)malloc(mc_ms_feat_size * sizeof(float));
    int* h_spatial_shape = (int*)malloc(spatial_shape_size * sizeof(int));
    int* h_scale_start_index = (int*)malloc(scale_start_index_size * sizeof(int));
    float* h_sample_location = (float*)malloc(sample_location_size * sizeof(float));
    float* h_weights = (float*)malloc(weights_size * sizeof(float));
    float* h_output = (float*)malloc(output_size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < mc_ms_feat_size; i++) {
        h_mc_ms_feat[i] = (float)(i % 100) / 100.0f;
    }
    for (int i = 0; i < spatial_shape_size; i++) {
        h_spatial_shape[i] = (i % 2 == 0) ? 4 : 4; // Example: 4x4 spatial shape
    }
    for (int i = 0; i < scale_start_index_size; i++) {
        h_scale_start_index[i] = i * 4; // Example: scale start indices
    }
    for (int i = 0; i < sample_location_size; i++) {
        h_sample_location[i] = (float)(i % 2) / 2.0f; // Example: normalized sample locations
    }
    for (int i = 0; i < weights_size; i++) {
        h_weights[i] = (float)(i % 5) / 5.0f; // Example: weights
    }

    // Allocate device memory
    float *d_mc_ms_feat, *d_sample_location, *d_weights, *d_output;
    int *d_spatial_shape, *d_scale_start_index;
    cudaMalloc(&d_mc_ms_feat, mc_ms_feat_size * sizeof(float));
    cudaMalloc(&d_spatial_shape, spatial_shape_size * sizeof(int));
    cudaMalloc(&d_scale_start_index, scale_start_index_size * sizeof(int));
    cudaMalloc(&d_sample_location, sample_location_size * sizeof(float));
    cudaMalloc(&d_weights, weights_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_mc_ms_feat, h_mc_ms_feat, mc_ms_feat_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_shape, h_spatial_shape, spatial_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_start_index, h_scale_start_index, scale_start_index_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sample_location, h_sample_location, sample_location_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Run the deformable aggregation function
    deformable_aggregation(
        d_output, d_mc_ms_feat, d_spatial_shape, d_scale_start_index,
        d_sample_location, d_weights, batch_size, num_cams, num_feat,
        num_embeds, num_scale, num_pts, num_groups
    );

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for deformable_aggregation kernel: %f ms\n", milliseconds);

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    printf("Output:\n");
    for (int i = 0; i < output_size; i++) {
        printf("%f ", h_output[i]);
        if ((i + 1) % num_embeds == 0) printf("\n");
    }

    // Free device memory
    cudaFree(d_mc_ms_feat);
    cudaFree(d_spatial_shape);
    cudaFree(d_scale_start_index);
    cudaFree(d_sample_location);
    cudaFree(d_weights);
    cudaFree(d_output);

    // Free host memory
    free(h_mc_ms_feat);
    free(h_spatial_shape);
    free(h_scale_start_index);
    free(h_sample_location);
    free(h_weights);
    free(h_output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_deformable_aggregation();
    return 0;
}




