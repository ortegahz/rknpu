// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cstdio>
#include <iostream>

#include <vector>

#define _BASETSD_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#undef cimg_display
#define cimg_display 0
#include "CImg/CImg.h"
#include "drm_func.h"
#include "postprocess.h"
#include "rga_func.h"

#define PERF_WITH_POST 0
#define SAVE_OUTPUTS 0
#define SAVE_F16_OUTPUTS 1
#define SAVE_F16_OUTPUTS_KPS 0

using namespace cimg_library;
/*-------------------------------------------
                  Functions
-------------------------------------------*/

void __f32_to_f16(uint16_t *f16, float *f32, int num)
{
  float *src = f32;
  uint16_t *dst = f16;
  int i = 0;

  for (; i < num; i++)
  {
    float in = *src;

    uint32_t fp32 = *((uint32_t *)&in);
    uint32_t t1 = (fp32 & 0x80000000u) >> 16; /* sign bit. */
    uint32_t t2 = (fp32 & 0x7F800000u) >> 13; /* Exponent bits */
    uint32_t t3 = (fp32 & 0x007FE000u) >> 13; /* Mantissa bits, no rounding */
    uint32_t fp16 = 0u;

    if (t2 >= 0x023c00u)
    {
      fp16 = t1 | 0x7BFF; /* Don't round to infinity. */
    }
    else if (t2 <= 0x01c000u)
    {
      fp16 = t1;
    }
    else
    {
      t2 -= 0x01c000u;
      fp16 = t1 | t2 | t3;
    }

    *dst = (uint16_t)fp16;

    src++;
    dst++;
  }
}

void __f16_to_f32(float *f32, uint16_t *f16, int num)
{
  uint16_t *src = f16;
  float *dst = f32;
  int i = 0;

  for (; i < num; i++)
  {
    uint16_t in = *src;

    int32_t t1;
    int32_t t2;
    int32_t t3;
    float out;

    t1 = in & 0x7fff; // Non-sign bits
    t2 = in & 0x8000; // Sign bit
    t3 = in & 0x7c00; // Exponent

    t1 <<= 13; // Align mantissa on MSB
    t2 <<= 16; // Shift sign bit into position

    t1 += 0x38000000; // Adjust bias

    t1 = (t3 == 0 ? 0 : t1); // Denormals-as-zero

    t1 |= t2; // Re-insert sign bit

    *((uint32_t *)&out) = t1;

    *dst = out;

    src++;
    dst++;
  }
}

static float __f16_to_f32_s(uint16_t f16)
{
  uint16_t in = f16;

  int32_t t1;
  int32_t t2;
  int32_t t3;
  uint32_t t4;
  float out;

  t1 = in & 0x7fff; // Non-sign bits
  t2 = in & 0x8000; // Sign bit
  t3 = in & 0x7c00; // Exponent

  t1 <<= 13; // Align mantissa on MSB
  t2 <<= 16; // Shift sign bit into position

  t1 += 0x38000000; // Adjust bias

  t1 = (t3 == 0 ? 0 : t1); // Denormals-as-zero

  t1 |= t2; // Re-insert sign bit

  *((uint32_t *)&out) = t1;

  return out;
}

inline const char *get_type_string(rknn_tensor_type type)
{
  switch (type)
  {
  case RKNN_TENSOR_FLOAT32:
    return "FP32";
  case RKNN_TENSOR_FLOAT16:
    return "FP16";
  case RKNN_TENSOR_INT8:
    return "INT8";
  case RKNN_TENSOR_UINT8:
    return "UINT8";
  case RKNN_TENSOR_INT16:
    return "INT16";
  default:
    return "UNKNOW";
  }
}

inline const char *get_qnt_type_string(rknn_tensor_qnt_type type)
{
  switch (type)
  {
  case RKNN_TENSOR_QNT_NONE:
    return "NONE";
  case RKNN_TENSOR_QNT_DFP:
    return "DFP";
  case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
    return "AFFINE";
  default:
    return "UNKNOW";
  }
}

inline const char *get_format_string(rknn_tensor_format fmt)
{
  switch (fmt)
  {
  case RKNN_TENSOR_NCHW:
    return "NCHW";
  case RKNN_TENSOR_NHWC:
    return "NHWC";
  default:
    return "UNKNOW";
  }
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
  FILE *fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++)
  {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

static unsigned char *load_image(const char *image_path, int *org_height, int *org_width, int *org_ch,
                                 rknn_tensor_attr *input_attr)
{
  int req_height = 0;
  int req_width = 0;
  int req_channel = 0;

  switch (input_attr->fmt)
  {
  case RKNN_TENSOR_NHWC:
    req_height = input_attr->dims[2];
    req_width = input_attr->dims[1];
    req_channel = input_attr->dims[0];
    break;
  case RKNN_TENSOR_NCHW:
    req_height = input_attr->dims[1];
    req_width = input_attr->dims[0];
    req_channel = input_attr->dims[2];
    break;
  default:
    printf("meet unsupported layout\n");
    return NULL;
  }

  int height = 0;
  int width = 0;
  int channel = 0;

  unsigned char *image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
  if (image_data == NULL)
  {
    printf("load image failed!\n");
    return NULL;
  }
  *org_width = width;
  *org_height = height;
  *org_ch = channel;

  return image_data;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
  int status = 0;
  char *model_name = NULL;
  rknn_context ctx;
  rknn_context ctx_kps;
  void *drm_buf = NULL;
  void *drm_buf_kps = NULL;
  int drm_fd = -1;
  int drm_fd_kps = -1;
  int buf_fd = -1;     // converted from buffer handle
  int buf_fd_kps = -1; // converted from buffer handle
  unsigned int handle;
  unsigned int handle_kps;
  size_t actual_size = 0;
  size_t actual_size_kps = 0;
  int img_width = 0;
  int img_height = 0;
  int img_channel = 0;
  int img_width_kps = 0;
  int img_height_kps = 0;
  int img_channel_kps = 0;
  rga_context rga_ctx;
  rga_context rga_ctx_kps;
  drm_context drm_ctx;
  drm_context drm_ctx_kps;
  const float nms_threshold = NMS_THRESH;
  const float box_conf_threshold = BOX_THRESH;
  struct timeval start_time, stop_time;
  int ret;
  memset(&rga_ctx, 0, sizeof(rga_context));
  memset(&drm_ctx, 0, sizeof(drm_context));

  if (argc != 3)
  {
    printf("Usage: %s <rknn model> <bmp> \n", argv[0]);
    return -1;
  }

  printf("post process config: box_conf_threshold = %f, nms_threshold = %f\n", box_conf_threshold, nms_threshold);

  model_name = (char *)argv[1];
  char *image_name = argv[2];

  if (strstr(image_name, ".jpg") != NULL || strstr(image_name, ".png") != NULL)
  {
    printf("Error: read %s failed! only support .bmp format image\n", image_name);
    return -1;
  }

  {
    /* create the kps neural network */
    printf("Loading kps mode...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model("./model/rv1109_rv1126/iter-96000.rknn", &model_data_size);
    ret = rknn_init(&ctx_kps, model_data, model_data_size, 0);
    if (ret < 0)
    {
      printf("kps rknn_init error ret=%d\n", ret);
      return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx_kps, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
      printf("kps rknn_query error ret=%d\n", ret);
      return -1;
    }
    printf("kps model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
      input_attrs[i].index = i;
      ret = rknn_query(ctx_kps, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
      if (ret < 0)
      {
        printf("kps rknn_query error ret=%d\n", ret);
        return -1;
      }
      dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
      output_attrs[i].index = i;
      ret = rknn_query(ctx_kps, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
      dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
      printf("kps model is NCHW input fmt\n");
      width = input_attrs[0].dims[0];
      height = input_attrs[0].dims[1];
    }
    else
    {
      printf("kps model is NHWC input fmt\n");
      width = input_attrs[0].dims[1];
      height = input_attrs[0].dims[2];
    }

    printf("kps model input height=%d, width=%d, channel=%d\n", height, width, channel);

    // rknn_input inputs[1];
    // memset(inputs, 0, sizeof(inputs));
    // inputs[0].index = 0;
    // inputs[0].type = RKNN_TENSOR_UINT8;
    // inputs[0].size = width * height * channel;
    // inputs[0].fmt = RKNN_TENSOR_NHWC;
    // inputs[0].pass_through = 0;

    // DRM alloc buffer
    // drm_fd_kps = drm_init(&drm_ctx_kps);
    // drm_buf_kps = drm_buf_alloc(&drm_ctx_kps, drm_fd_kps, img_width_kps, img_height_kps, channel * 8, &buf_fd_kps, &handle_kps, &actual_size_kps);
    // memcpy(drm_buf_kps, input_data, img_width_kps * img_height_kps * channel);
    void *resize_buf = malloc(height * width * channel);
    // unsigned char *p = (unsigned char *) resize_buf;

    // cv::Mat Img = cv::imread("./model/rsn.bmp");
    // pcBOX_RECT_FLOAT stBoxRect = {0};
    // stBoxRect.left = 153.53;
    // stBoxRect.top = 231.12;
    // stBoxRect.right = stBoxRect.left + 270.17;
    // stBoxRect.bottom = stBoxRect.top + 403.95;

    cv::Mat Img = cv::imread("./model/player_1280.bmp");
    pcBOX_RECT_FLOAT stBoxRect = {0};
    stBoxRect.left = 825.;
    stBoxRect.top = 679.;
    stBoxRect.right = stBoxRect.left + 111.1;
    stBoxRect.bottom = stBoxRect.top + 244.2;

    kps_result_group_t kps_result_group;

    post_process_kps_f16_wrapper(ctx_kps, &Img, stBoxRect, resize_buf, output_attrs, &kps_result_group);

    // init rga context
    // RGA_init(&rga_ctx_kps);
    // img_resize_slow_kps(&rga_ctx_kps, drm_buf_kps, img_width_kps, img_height_kps, resize_buf, width, height);

    // memcpy(resize_buf, input_data, height * width * channel);

    // cv::Mat img_save = img;
    // img_save.data = (unsigned char *) resize_buf;
    // cv::imwrite("./img_save.bmp", img_save);

    // inputs[0].buf = resize_buf;
    // gettimeofday(&start_time, NULL);
    // rknn_inputs_set(ctx_kps, io_num.n_input, inputs);

    // rknn_output outputs[io_num.n_output];
    // memset(outputs, 0, sizeof(outputs));
    // for (int i = 0; i < io_num.n_output; i++)
    // {
    //   outputs[i].want_float = 0;
    // }

    // ret = rknn_run(ctx_kps, NULL);
    // ret = rknn_outputs_get(ctx_kps, io_num.n_output, outputs, NULL);
    // gettimeofday(&stop_time, NULL);
    // printf("kps once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // float scale_w = (float)width / img_width_kps;
    // float scale_h = (float)height / img_height_kps;

    // kps_result_group_t kps_result_group;
    // std::vector<float> out_scales;
    // std::vector<uint32_t> out_zps;
    // for (int i = 0; i < io_num.n_output; ++i)
    // {
    //   out_scales.push_back(output_attrs[i].scale);
    //   out_zps.push_back(output_attrs[i].zp);
    // }

// #if SAVE_F16_OUTPUTS_KPS
//     // save float outputs for debugging
//     for (int i = 0; i < io_num.n_output; ++i)
//     {
//       char path[512];
//       sprintf(path, "./rknn_output_real_kps_nq_%d.txt", i);
//       FILE *fp = fopen(path, "w");
//       uint16_t *output = (uint16_t *)outputs[i].buf;
//       uint32_t n_elems = output_attrs[i].n_elems;
//       for (int j = 0; j < n_elems; j++)
//       {
//         float value = __f16_to_f32_s(output[j]);
//         fprintf(fp, "%f\n", value);
//       }
//       fclose(fp);
//     }
// #endif

  // post_process_kps_f16((uint16_t *)outputs[0].buf, &kps_result_group);

  // // Save KPS Parser Results
  // FILE * fid = fopen("npu_parser_results_kps.txt", "w");
  // assert(fid != NULL);
  // for (int i = 0; i < kps_result_group.count; i++) {
  //   kps_result_t* kps_result = &(kps_result_group.results[i]);
  //   for (int j = 0; j < KPS_KEYPOINT_NUM; j++) {
  //     float x = (float) kps_result->kps[j].x;
  //     float y = (float) kps_result->kps[j].y;
  //     float conf = kps_result->kps[j].conf;
  //     fprintf(fid, "%f, %f,  %f \n", x, y, conf);
  //   }
  // }
  // fclose(fid);

    // release
    // ret = rknn_destroy(ctx_kps);
    // drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
    // drm_deinit(&drm_ctx, drm_fd);
    // RGA_deinit(&rga_ctx);
    if (model_data)
    {
      free(model_data);
    }

    if (resize_buf)
    {
      free(resize_buf);
    }
    // stbi_image_free(input_data);

    // return 0;
  }

  /* Create the neural network */
  printf("Loading mode...\n");
  int model_data_size = 0;
  unsigned char *model_data = load_model(model_name, &model_data_size);
  ret = rknn_init(&ctx, model_data, model_data_size, 0);
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }

  rknn_sdk_version version;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++)
  {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
    printf("output_attrs[%d].type --> %d \n", i, output_attrs[i].type);

    // if (output_attrs[i].qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC || output_attrs[i].type != RKNN_TENSOR_UINT8) {
    //   fprintf(stderr,
    //           "The Demo required for a Affine asymmetric u8 quantized rknn model, but output quant type is %s, output "
    //           "data type is %s\n",
    //           get_qnt_type_string(output_attrs[i].qnt_type), get_type_string(output_attrs[i].type));
    //   return -1;
    // }
  }

  int channel = 3;
  int width = 0;
  int height = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
  {
    printf("model is NCHW input fmt\n");
    width = input_attrs[0].dims[0];
    height = input_attrs[0].dims[1];
  }
  else
  {
    printf("model is NHWC input fmt\n");
    width = input_attrs[0].dims[1];
    height = input_attrs[0].dims[2];
  }

  printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  // Load image
  CImg<unsigned char> img(image_name);
  unsigned char *input_data = NULL;
  input_data = load_image(image_name, &img_height, &img_width, &img_channel, &input_attrs[0]);
  if (!input_data)
  {
    return -1;
  }

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = width * height * channel;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;

  // DRM alloc buffer
  drm_fd = drm_init(&drm_ctx);
  drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, img_width, img_height, channel * 8, &buf_fd, &handle, &actual_size);
  memcpy(drm_buf, input_data, img_width * img_height * channel);
  void *resize_buf = malloc(height * width * channel);

  // init rga context
  RGA_init(&rga_ctx);
  img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, width, height);
  inputs[0].buf = resize_buf;
  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx, io_num.n_input, inputs);

  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++)
  {
    outputs[i].want_float = 0;
  }

  ret = rknn_run(ctx, NULL);
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  gettimeofday(&stop_time, NULL);
  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  // post process
  float scale_w = (float)width / img_width;
  float scale_h = (float)height / img_height;

  detect_result_group_float_t detect_result_group;
  std::vector<float> out_scales;
  std::vector<uint32_t> out_zps;
  for (int i = 0; i < io_num.n_output; ++i)
  {
    out_scales.push_back(output_attrs[i].scale);
    out_zps.push_back(output_attrs[i].zp);
  }

#if SAVE_OUTPUTS
  // save float outputs for debugging
  for (int i = 0; i < io_num.n_output; ++i)
  {
    char path[128];
    sprintf(path, "./rknn_output_real_%d.txt", i);
    FILE *fp = fopen(path, "w");
    uint8_t *output = (uint8_t *)outputs[i].buf;
    float out_scale = output_attrs[i].scale;
    uint32_t out_zp = output_attrs[i].zp;
    uint32_t n_elems = output_attrs[i].n_elems;
    // printf("output idx %d n_elems --> %d \n", i, n_elems);
    for (int j = 0; j < n_elems; j++)
    {
      float value = deqnt_affine_to_f32(output[j], out_zp, out_scale);
      fprintf(fp, "%f\n", value);
    }
    fclose(fp);
  }
#endif

#if SAVE_F16_OUTPUTS
  // save float outputs for debugging
  for (int i = 0; i < io_num.n_output; ++i)
  {
    char path[128];
    sprintf(path, "./rknn_output_real_nq_%d.txt", i);
    FILE *fp = fopen(path, "w");
    uint16_t *output = (uint16_t *)outputs[i].buf;
    uint32_t n_elems = output_attrs[i].n_elems;
    for (int j = 0; j < n_elems; j++)
    {
      float value = __f16_to_f32_s(output[j]);
      fprintf(fp, "%f\n", value);
    }
    fclose(fp);
  }
#endif

  // post_process((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, height, width,
  //              box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
  // post_process_acfree((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, (uint8_t*)outputs[3].buf, (uint8_t*)outputs[4].buf, (uint8_t*)outputs[5].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
  // post_process_acfree_f16((uint16_t*)outputs[0].buf, (uint16_t*)outputs[1].buf, (uint16_t*)outputs[2].buf, (uint16_t*)outputs[3].buf, (uint16_t*)outputs[4].buf, (uint16_t*)outputs[5].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, &detect_result_group);
  // post_process_acfree_6_f16((uint16_t*)outputs[0].buf, (uint16_t*)outputs[1].buf, (uint16_t*)outputs[2].buf, (uint16_t*)outputs[3].buf, (uint16_t*)outputs[4].buf, (uint16_t*)outputs[5].buf, (uint16_t*)outputs[6].buf, (uint16_t*)outputs[7].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, &detect_result_group);
  // post_process_acfree_6((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, (uint8_t*)outputs[3].buf, (uint8_t*)outputs[4].buf, (uint8_t*)outputs[5].buf, (uint8_t*)outputs[6].buf, (uint8_t*)outputs[7].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
  post_process_player_6_f16((uint16_t*)outputs[0].buf, (uint16_t*)outputs[1].buf, (uint16_t*)outputs[2].buf, (uint16_t*)outputs[3].buf, (uint16_t*)outputs[4].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, &detect_result_group);
  // post_process_player_6((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, (uint8_t *)outputs[3].buf, (uint8_t *)outputs[4].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

  {
    // Draw Objects
    char text[256];
    const unsigned char blue[] = {0, 0, 255};
    const unsigned char red[] = {255, 0, 0};
    const unsigned char white[] = {255, 255, 255};
    for (int i = 0; i < detect_result_group.count; i++)
    {
      detect_result_float_t *det_result = &(detect_result_group.results[i]);
      sprintf(text, "%s %.2f", det_result->name, det_result->prop);
      // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
      //        det_result->box.right, det_result->box.bottom, det_result->prop);
      float x1 = det_result->box.left;
      float y1 = det_result->box.top;
      float x2 = det_result->box.right;
      float y2 = det_result->box.bottom;
      // draw box
      img.draw_rectangle(x1, y1, x2, y2, red, 1, ~0U);
      img.draw_text(x1, y1 - 12, text, white);
      float xc = (x1 + x2) / 2;
      float yc = (y1 + y2) / 2;
      float x = det_result->poi.x;
      float y = det_result->poi.y;
      float conf = det_result->poi.conf;
      // printf("pp_x pp_y pp_c --> %d, %d, %f\n", x, y, conf);
      sprintf(text, "%.2f", conf);
      img.draw_line(xc, yc, x, y, blue);
      img.draw_text(xc, yc, text, blue);
    }
    img.save("./out.bmp");
  }

  {
    // Save Parser Results
    FILE *fid = fopen("npu_parser_results.txt", "w");
    assert(fid != NULL);
    for (int i = 0; i < detect_result_group.count; i++)
    {
      detect_result_float_t *det_result = &(detect_result_group.results[i]);
      // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
      //        det_result->box.right, det_result->box.bottom, det_result->prop);
      float x1 = det_result->box.left;
      float y1 = det_result->box.top;
      float x2 = det_result->box.right;
      float y2 = det_result->box.bottom;
      float xc = (x1 + x2) / 2 / (float)width;
      float yc = (y1 + y2) / 2 / (float)height;
      float w = (x2 - x1) / (float)width;
      float h = (y2 - y1) / (float)height;
      int x = det_result->poi.x;
      int y = det_result->poi.y;
      float conf = det_result->poi.conf;
      // TODO: auto class id
      fprintf(fid, "0, %f, %f,  %f, %f, %d, %d, %f, %f\n", xc, yc, w, h, x, y, conf, det_result->prop);
    }
    fclose(fid);
  }

  {
    cv::Mat Img = cv::imread("./model/player_1280.bmp");
    FILE *pFileHandle = fopen("npu_parser_final_results.txt", "w");
    assert(pFileHandle != NULL);
    for (int i = 0; i < detect_result_group.count; i++)
    {
      detect_result_float_t *det_result = &(detect_result_group.results[i]);
      float x1 = det_result->box.left;
      float y1 = det_result->box.top;
      float x2 = det_result->box.right;
      float y2 = det_result->box.bottom;
      float xc = (x1 + x2) / 2 / (float)width;
      float yc = (y1 + y2) / 2 / (float)height;
      float w = (x2 - x1) / (float)width;
      float h = (y2 - y1) / (float)height;
      int x = det_result->poi.x;
      int y = det_result->poi.y;
      float conf = det_result->poi.conf;

      void *resize_buf = malloc(height * width * channel);
      // unsigned char *p = (unsigned char *) resize_buf;

      // float fXC = (x1 + x2) / 2;
      // float fYC = (y1 + y2) / 2;
      float fWE = (x2 - x1) * KPS_W_EXTENTION;
      float fHE = (y2 - y1) * KPS_H_EXTENTION;
      pcBOX_RECT_FLOAT stBoxRect = {0};
      stBoxRect.left = x1;
      stBoxRect.top = y1;
      stBoxRect.right = x1 + fWE / 2.;
      stBoxRect.bottom = y1 + fHE / 2.;

      // stBoxRect.left = 825.;
      // stBoxRect.top = 679.;
      // stBoxRect.right = stBoxRect.left + 111.1;
      // stBoxRect.bottom = stBoxRect.top + 244.2;

      printf("[i] stBoxRect.left, stBoxRect.top, fWE, fHE --> %d, %f, %f, %f, %f \n", i, stBoxRect.left, stBoxRect.top, fWE, fHE);
      // return 0;

      kps_result_group_t kps_result_group;

      post_process_kps_f16_wrapper(ctx_kps, &Img, stBoxRect, resize_buf, output_attrs, &kps_result_group);

      fprintf(pFileHandle, "0, %f, %f,  %f, %f, %d, %d, %f, %f ", xc, yc, w, h, x, y, conf, det_result->prop);
      for (int j = 0; j < KPS_KEYPOINT_NUM; j++) {
        float x = (float) kps_result_group.results->kps[j].x;
        float y = (float) kps_result_group.results->kps[j].y;
        float conf = kps_result_group.results->kps[j].conf;
        fprintf(pFileHandle, "%f,  %f, %f ", x, y, conf);
      }
      fprintf(pFileHandle, "\n");
    }
    fclose(pFileHandle);
  }

  ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

  // loop test
  int test_count = 0;
  gettimeofday(&start_time, NULL);
  for (int i = 0; i < test_count; ++i)
  {
    // img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, width, height);
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
#if PERF_WITH_POST
    // post_process((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, height, width,
    //              box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    post_process_acfree((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, (uint8_t *)outputs[3].buf, (uint8_t *)outputs[4].buf, (uint8_t *)outputs[5].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    // post_process_acfree_f16((uint16_t*)outputs[0].buf, (uint16_t*)outputs[1].buf, (uint16_t*)outputs[2].buf, (uint16_t*)outputs[3].buf, (uint16_t*)outputs[4].buf, (uint16_t*)outputs[5].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, &detect_result_group);
#endif
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
  }
  gettimeofday(&stop_time, NULL);
  printf("loop count = %d , average run  %f ms\n", test_count,
         (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

  // release
  ret = rknn_destroy(ctx);
  ret = rknn_destroy(ctx_kps);
  drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);

  drm_deinit(&drm_ctx, drm_fd);
  RGA_deinit(&rga_ctx);
  if (model_data)
  {
    free(model_data);
  }

  if (resize_buf)
  {
    free(resize_buf);
  }
  stbi_image_free(input_data);

  return 0;
}
