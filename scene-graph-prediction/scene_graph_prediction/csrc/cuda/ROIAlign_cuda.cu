// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>

// c_todo make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data, const int height, const int width, T y, T x) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename scalar_t>
__device__ scalar_t trilinear_interpolate(const scalar_t *bottom_data,
                                          const int depth, const int height, const int width,
                                          scalar_t z, scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > depth || y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int z_low = (int)z;
  int y_low = (int)y;
  int x_low = (int)x;
  int z_high;
  int y_high;
  int x_high;

  if (z_low >= depth - 1) {
    z_high = z_low = depth - 1;
    z = (scalar_t)z_low;
  } else {
    z_high = z_low + 1;
  }

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t lz = z - z_low;
  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hz = 1. - lz;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  // based on https://github.com/pfjaeger/medicaldetectiontoolkit/blob/master/cuda_functions/roi_align_3D/roi_align/src/cuda/crop_and_resize_kernel.cu
  // do trilinear interpolation

  const float top_left_front = bottom_data[x_low + width * (y_low + height * z_low)];
  const float top_right_front = bottom_data[x_high + width * (y_low + height * z_low)];
  const float bottom_left_front = bottom_data[x_low + width * (y_high + height * z_low)];
  const float bottom_right_front = bottom_data[x_high + width * (y_high + height * z_low)];
  const float top_left_back = bottom_data[x_low + width * (y_low + height * z_high)];
  const float top_right_back = bottom_data[x_high + width * (y_low + height * z_high)];
  const float bottom_left_back = bottom_data[x_low + width * (y_high + height * z_high)];
  const float bottom_right_back = bottom_data[x_high + width * (y_high + height * z_high)];

  scalar_t w1 = hx * hy * hz; scalar_t w5 = hx * hy * lz;
  scalar_t w2 = lx * hy * hz; scalar_t w6 = lx * hy * lz;
  scalar_t w3 = hx * ly * hz; scalar_t w7 = hx * ly * lz;
  scalar_t w4 = lx * ly * hz; scalar_t w8 = lx * ly * lz;

  scalar_t val = (w1 * top_left_front +
                  w2 * top_right_front +
                  w3 * bottom_left_front +
                  w4 * bottom_right_front +
                  w5 * top_left_back +
                  w6 * top_right_back +
                  w7 * bottom_left_back +
                  w8 * bottom_right_back);

  return val;
}


template <typename T>
__global__ void RoIAlignForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_start_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;
    T roi_end_w = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w + 1, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h + 1, (T)1.);

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(offset_bottom_data, height, width, y, x);
        output_val += val;
      }
    }

    // We do average (integral) pooling inside a bin
    output_val /= roi_bin_grid_h * roi_bin_grid_w;

    top_data[index] = output_val;
  }
}


template <typename scalar_t>
__global__ void RoIAlignForward3D(const int nthreads, const scalar_t *bottom_data,
                                  const scalar_t spatial_scale, const scalar_t spatial_scale_depth,
                                  const int channels, const int depth, const int height, const int width,
                                  const int pooled_depth, const int pooled_height, const int pooled_width,
                                  const int sampling_ratio,
                                  const scalar_t *bottom_rois, scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, depth, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_width / pooled_height) % pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height / pooled_depth / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];

    scalar_t roi_start_d = offset_bottom_rois[1] * spatial_scale_depth;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_start_w = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_d = offset_bottom_rois[4] * spatial_scale_depth;
    scalar_t roi_end_h = offset_bottom_rois[5] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[6] * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w + 1, 1.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h + 1, 1.);
    scalar_t roi_depth = fmaxf((scalar_t)roi_end_d - roi_start_d + 1, 1.);

    scalar_t bin_size_d = (scalar_t)(roi_depth) / (scalar_t)(pooled_depth);
    scalar_t bin_size_h = (scalar_t)(roi_height) / (scalar_t)(pooled_height);
    scalar_t bin_size_w = (scalar_t)(roi_width) / (scalar_t)(pooled_width);

    const scalar_t *offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * depth * height * width;

    int sampling_ratio_d = (sampling_ratio > 0) 
                           ? sampling_ratio 
                           : ceil(roi_depth / pooled_depth);
    int sampling_ratio_h = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sampling_ratio_w = (sampling_ratio > 0) 
                           ? sampling_ratio 
                           : ceil(roi_width / pooled_width);

    scalar_t d = (scalar_t)(pd + 0.5) * bin_size_d + roi_start_d;
    scalar_t h = (scalar_t)(ph + 0.5) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw + 0.5) * bin_size_w + roi_start_w;

    scalar_t output_val = 0;
    for (int iz = 0; iz < sampling_ratio_d; iz++) {
      const scalar_t z = roi_start_d + pd * bin_size_d +
                         (scalar_t)(iz) * bin_size_d /
                             (scalar_t)(sampling_ratio_d);
      for (int iy = 0; iy < sampling_ratio_h; iy++) {
        const scalar_t y = roi_start_h + ph * bin_size_h +
                          (scalar_t)(iy) * bin_size_h /
                              (scalar_t)(sampling_ratio_h);
        for (int ix = 0; ix < sampling_ratio_w; ix++) {
          const scalar_t x = roi_start_w + pw * bin_size_w +
                            (scalar_t)(ix) * bin_size_w /
                                (scalar_t)(sampling_ratio_w);
          scalar_t val = trilinear_interpolate<scalar_t>(offset_bottom_data, depth, height, width, z, y, x);
          output_val += val;
        }
      }
    }

    output_val /= sampling_ratio_d * sampling_ratio_h * sampling_ratio_w;
    top_data[index] = output_val;
  }
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
__device__ void trilinear_interpolate_gradient(const int depth, const int height, const int width,
                                               scalar_t z, scalar_t y, scalar_t x,
                                               scalar_t &w1, scalar_t &w2,
                                               scalar_t &w3, scalar_t &w4,
                                               scalar_t &w5, scalar_t &w6,
                                               scalar_t &w7, scalar_t &w8,
                                               int &x_low, int &x_high,
                                               int &y_low, int &y_high,
                                               int &z_low, int &z_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > depth || y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
    x_low = x_high = y_low = y_high = z_low = z_high = -1;
    return;
  }

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  z_low = (int) z;
  y_low = (int) y;
  x_low = (int) x;

  if (z_low >= depth - 1) {
    z_high = z_low = depth - 1;
    z = (scalar_t)z_low;
  } else {
    z_high = z_low + 1;
  }

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t lz = z - z_low;
  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hz = 1. - lz;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  w1 = hx * hy * hz; w5 = hx * hy * lz;
  w2 = lx * hy * hz; w6 = lx * hy * lz;
  w3 = hx * ly * hz; w7 = hx * ly * lz;
  w4 = lx * ly * hz; w8 = lx * ly * lz;

  return;
}


template <typename T>
__global__ void RoIAlignBackwardFeature(
    const int nthreads, const T* top_diff,
    const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w + 1, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h + 1, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height, width, y, x,
            w1, w2, w3, w4,
            x_low, x_high, y_low, y_high);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
        {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
          atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
          atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
          atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward


template <typename scalar_t>
__global__ void RoIAlignBackwardFeature3D(
    const int nthreads, const scalar_t *top_diff,
    const scalar_t spatial_scale, const scalar_t spatial_scale_depth,
    const int channels, const int depth, const int height, const int width,
    const int pooled_depth, const int pooled_height, const int pooled_width,
    const int sample_num,
    scalar_t *bottom_diff,
    const scalar_t *bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pd, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_width / pooled_height) % pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height / pooled_depth / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_d = offset_bottom_rois[1] * spatial_scale_depth;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_start_w = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_d = offset_bottom_rois[4] * spatial_scale_depth;
    scalar_t roi_end_h = offset_bottom_rois[5] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[6] * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w + 1, 1.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h + 1, 1.);
    scalar_t roi_depth = fmaxf((scalar_t)roi_end_d - roi_start_d + 1, 1.);

    scalar_t bin_size_d = roi_depth / pooled_depth;
    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;

    scalar_t *offset_bottom_diff =
      bottom_diff + (roi_batch_ind * channels + c) * depth * height * width;

    // Note: there might have been an error here
    int top_offset = (n * channels + c) * pooled_depth * pooled_height * pooled_width;
    const scalar_t *offset_top_diff_tmp = top_diff + top_offset;
    scalar_t offset_top_diff = offset_top_diff_tmp[pd * pooled_height * pooled_width + ph * pooled_width + pw];

    int sample_num_d = (sample_num > 0) ? sample_num : ceil(roi_depth / pooled_depth);
    int sample_num_h = (sample_num > 0) ? sample_num : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w = (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    const scalar_t count = (scalar_t)(sample_num_d * sample_num_h * sample_num_w);

    scalar_t d = (scalar_t)(pd + 0.5) * bin_size_d + roi_start_d;
    scalar_t h = (scalar_t)(ph + 0.5) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw + 0.5) * bin_size_w + roi_start_w;

    for (int iz = 0; iz < sample_num_d; iz++) {
      const scalar_t z =
          roi_start_d + pd * bin_size_d +
          (scalar_t)(iz) * bin_size_d / (scalar_t)(sample_num_d);
      for (int iy = 0; iy < sample_num_h; iy++) {
        const scalar_t y =
            roi_start_h + ph * bin_size_h +
            (scalar_t)(iy) * bin_size_h / (scalar_t)(sample_num_h);
        for (int ix = 0; ix < sample_num_w; ix++) {
          const scalar_t x =
              roi_start_w + pw * bin_size_w +
              (scalar_t)(ix) * bin_size_w / (scalar_t)(sample_num_w);
          scalar_t w1, w2, w3, w4, w5, w6, w7, w8;
          int x_low, x_high, y_low, y_high, z_low, z_high;

          trilinear_interpolate_gradient<scalar_t>(
              depth, height, width, z, y, x, w1, w2, w3, w4, w5, w6, w7, w8, x_low, x_high, y_low, y_high, z_low, z_high);
          scalar_t g1 = offset_top_diff * w1 / count;
          scalar_t g2 = offset_top_diff * w2 / count;
          scalar_t g3 = offset_top_diff * w3 / count;
          scalar_t g4 = offset_top_diff * w4 / count;

          scalar_t g5 = offset_top_diff * w5 / count;
          scalar_t g6 = offset_top_diff * w6 / count;
          scalar_t g7 = offset_top_diff * w7 / count;
          scalar_t g8 = offset_top_diff * w8 / count;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && z_low >= 0 && z_high >= 0) {
            atomicAdd(offset_bottom_diff + (z_low * height + y_low) * width + x_low, g1);    // top_left_front
            atomicAdd(offset_bottom_diff + (z_low * height + y_low) * width + x_high, g2);   // top_right_front
            atomicAdd(offset_bottom_diff + (z_low * height + y_high) * width + x_low, g3);   // bottom_left_front
            atomicAdd(offset_bottom_diff + (z_low * height + y_high) * width + x_high, g4);  // bottom_right_front

            atomicAdd(offset_bottom_diff + (z_high * height + y_low) * width + x_low, g5);   // top_left_back
            atomicAdd(offset_bottom_diff + (z_high * height + y_low) * width + x_high, g6);  // top_right_back
            atomicAdd(offset_bottom_diff + (z_high * height + y_high) * width + x_low, g7);  // bottom_left_back
            atomicAdd(offset_bottom_diff + (z_high * height + y_high) * width + x_high, g8); // bottom_right_back
          }
        }
      }
    }
  }
}


at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio) {
  AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::ceil_div((long)output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ROIAlign_forward", [&] {
    RoIAlignForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data_ptr<scalar_t>(),
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         sampling_ratio,
         rois.contiguous().data_ptr<scalar_t>(),
         output.data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

at::Tensor ROIAlign_forward_cuda_3d(const at::Tensor& input,
                                    const at::Tensor& rois,
                                    const float spatial_scale,
                                    const float spatial_scale_depth,
                                    const int pooled_depth,
                                    const int pooled_height,
                                    const int pooled_width,
                                    const int sampling_ratio) {
  AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto depth = input.size(2);
  auto height = input.size(3);
  auto width = input.size(4);

  auto output = at::empty({num_rois, channels, pooled_depth, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_depth * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }


  dim3 grid(std::min(at::ceil_div((long)output_size, 512L), 4096L));
  dim3 block(512);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ROIAlign_forward", [&] {
    RoIAlignForward3D<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data_ptr<scalar_t>(),
         spatial_scale,
         spatial_scale_depth,
         channels,
         depth,
         height,
         width,
         pooled_depth,
         pooled_height,
         pooled_width,
         sampling_ratio,
         rois.contiguous().data_ptr<scalar_t>(),
         output.data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio) {
  AT_ASSERTM(grad.is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::ceil_div((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "ROIAlign_backward", [&] {
    RoIAlignBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data_ptr<scalar_t>(),
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         sampling_ratio,
         grad_input.data_ptr<scalar_t>(),
         rois.contiguous().data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}

at::Tensor ROIAlign_backward_cuda_3d(const at::Tensor& grad,
                                     const at::Tensor& rois,
                                     const float spatial_scale,
                                     const float spatial_scale_depth,
                                     const int pooled_depth,
                                     const int pooled_height,
                                     const int pooled_width,
                                     const int batch_size,
                                     const int channels,
                                     const int depth,
                                     const int height,
                                     const int width,
                                     const int sampling_ratio) {
  AT_ASSERTM(grad.is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, depth, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::ceil_div((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "ROIAlign_backward", [&] {
    RoIAlignBackwardFeature3D<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data_ptr<scalar_t>(),
         spatial_scale,
         spatial_scale_depth,
         channels,
         depth,
         height,
         width,
         pooled_depth,
         pooled_height,
         pooled_width,
         sampling_ratio,
         grad_input.data_ptr<scalar_t>(),
         rois.contiguous().data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}
