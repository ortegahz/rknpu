#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 1024
#define OBJ_CLASS_NUM     1
#define NMS_THRESH        0.3
#define BOX_THRESH        0.4
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process_acfree_f16(uint16_t* input0, uint16_t* input1, uint16_t* input2, uint16_t* input3, uint16_t* input4, uint16_t* input5, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, detect_result_group_t* group);

int post_process_acfree(uint8_t* input0, uint8_t* input1, uint8_t* input2, uint8_t* input3, uint8_t* input4, uint8_t* input5, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<uint32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group);

int post_process(uint8_t *input0, uint8_t *input1, uint8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

float deqnt_affine_to_f32(uint8_t qnt, uint32_t zp, float scale);

#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
