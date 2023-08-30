#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "rknn_api.h"

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 1024
#define OBJ_CLASS_NUM     1
#define NMS_THRESH        0.3
#define BOX_THRESH        0.4
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

#define KPS_PIXEL_STD 200
#define KPS_PIXEL_BORDER 10
#define KPS_GAUSSIAN_KERNEL 5
#define KPS_KEYPOINT_NUM 17
#define KPS_INPUT_SHAPE_H 256
#define KPS_INPUT_SHAPE_W 192
#define KPS_WIDTH_HEIGHT_RATIO ((float) KPS_INPUT_SHAPE_W / (float) KPS_INPUT_SHAPE_H)
#define KPS_OUTPUT_SHAPE_H 64
#define KPS_OUTPUT_SHAPE_W 48
#define KPS_STRIDE 4
#define KPS_SHIFTS 0.25
#define KPS_X_EXTENTION (0.01 * 9.0)
#define KPS_Y_EXTENTION (0.015 * 9.0)
#define KPS_W_EXTENTION 1.1
#define KPS_H_EXTENTION 2.2

typedef struct _POI_FLOAT
{
    float x;
    float y;
    float conf;
} POI_FLOAT;

typedef struct _POI
{
    int x;
    int y;
    float conf;
} POI;

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct pcBOX_RECT_FLOAT
{
    float left;
    float right;
    float top;
    float bottom;
} BOX_RECT_FLOAT;

typedef struct _detect_result_float_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT_FLOAT box;
    float prop;
    POI_FLOAT poi;
} detect_result_float_t;

typedef struct _detect_result_group_float_t
{
    int id;
    int count;
    detect_result_float_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_float_t;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
    POI poi;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

typedef struct _KP
{
    float x;
    float y;
    float conf;
} KP;

typedef struct _kps_result_t
{
    KP kps[KPS_KEYPOINT_NUM];
} kps_result_t;

typedef struct _kps_result_group_t
{
    int count;
    kps_result_t results[OBJ_NUMB_MAX_SIZE];
} kps_result_group_t;

int post_process_kps_f16_wrapper(rknn_context ctx_kps, cv::Mat *Img, pcBOX_RECT_FLOAT stBoxRect, void *resize_buf, rknn_tensor_attr *output_attrs, kps_result_group_t *pKps_result_group);

int post_process_kps_f16(uint16_t *pu16Input, float fCenterX, float fCenterY, float fScaleWT, float fScaleHT, kps_result_group_t *group);

int post_process_player_6(uint8_t* input0, uint8_t* input1, uint8_t* input2, uint8_t* input3, uint8_t* input4, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, std::vector<uint32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_t* group);

int post_process_player_6_f16(uint16_t* input0, uint16_t* input1, uint16_t* input2, uint16_t* input3, uint16_t* input4, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, detect_result_group_float_t* group);

int post_process_acfree_6_f16(uint16_t* input0, uint16_t* input1, uint16_t* input2, uint16_t* input3, uint16_t* input4, uint16_t* input5, uint16_t* input6, uint16_t* input7, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, detect_result_group_t* group);

int post_process_acfree_f16(uint16_t* input0, uint16_t* input1, uint16_t* input2, uint16_t* input3, uint16_t* input4, uint16_t* input5, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, detect_result_group_t* group);

int post_process_acfree(uint8_t* input0, uint8_t* input1, uint8_t* input2, uint8_t* input3, uint8_t* input4, uint8_t* input5, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<uint32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group);

int post_process(uint8_t *input0, uint8_t *input1, uint8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

int post_process_acfree_6(uint8_t* input0, uint8_t* input1, uint8_t* input2, uint8_t* input3, uint8_t* input4, uint8_t* input5, uint8_t* input6, uint8_t* input7, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<uint32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group);

float deqnt_affine_to_f32(uint8_t qnt, uint32_t zp, float scale);

#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
