/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License */

#include <iostream>
#include <sstream>
#include <string.h>
#include <paddle/capi.h>
#include <cmath>

#include "frcnn_reader.h"

static const char* paddle_error_string(paddle_error status) {
  switch (status) {
    case kPD_NULLPTR:
      return "nullptr error";
    case kPD_OUT_OF_RANGE:
      return "out of range error";
    case kPD_PROTOBUF_ERROR:
      return "protobuf error";
    case kPD_NOT_SUPPORTED:
      return "not supported error";
    case kPD_UNDEFINED_ERROR:
      return "undefined error";
  };
}

#define CHECK_PD(stmt)                                         \
  do {                                                         \
    paddle_error __err__ = stmt;                               \
    if (__err__ != kPD_NO_ERROR) {                             \
      const char* str = paddle_error_string(__err__);          \
      fprintf(stderr, "%s (%d) in " #stmt "\n", str, __err__); \
      exit(__err__);                                           \
    }                                                          \
  } while (0)



#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    exit(EXIT_FAILURE);                                                \
}

#define CHECK(flag)                                            \
  do {                                                         \
    if (!flag) {                                               \
      FatalError("Error");                                     \
    }                                                          \
  } while (0)


static void* read_config(const char* filename, long* size) {
  FILE* file = fopen(filename, "rb");
  if (file == NULL) {
    fprintf(stderr, "Open %s error\n", filename);
    return NULL;
  }
  fseek(file, 0L, SEEK_END);
  *size = ftell(file);
  fseek(file, 0L, SEEK_SET);
  void* buf = malloc(*size);
  fread(buf, 1, *size, file);
  fclose(file);
  return buf;
}

void init_paddle() {
  // Initalize Paddle
  static bool called = false;
  if (!called) {
    char* argv[] = {const_cast<char*>("--use_gpu=False")};
    CHECK_PD(paddle_init(1, (char**)argv));
    called = true;
  }
}

paddle_gradient_machine init(const char* merged_model_path) {
  // Step 1: Reading merged model.
  long size;
  void* buf = read_config(merged_model_path, &size);

  // Step 2:
  //    Create a gradient machine for inference.
  paddle_gradient_machine machine;
  CHECK_PD(paddle_gradient_machine_create_for_inference_with_parameters(
      &machine, buf, size));

  free(buf);
  buf = nullptr;
  return machine;
}

void infer(paddle_gradient_machine machine,
           const FrcnnData* data,
           float*& result,
           uint64_t* result_height,
           uint64_t* result_width) {

  CHECK(data->input.size() == 1UL);
  CHECK(data->roi.size() == 1UL);
  CHECK(data->prob.size() == 1UL);
  CHECK(data->bbox.size() == 1UL);

  // Step 3: Prepare input Arguments.
  paddle_arguments in_args = paddle_arguments_create_none();

  // There are two inputs: (input image, rois)
  CHECK_PD(paddle_arguments_resize(in_args, 2));

  // Create first input
  paddle_matrix mat = paddle_matrix_create(
      /* sample_num */ 1,
      /* size */ data->input[0].size,
      /* useGPU */ false);
  CHECK_PD(paddle_matrix_set_value(mat, data->input[0].data));
  CHECK_PD(paddle_arguments_set_value(in_args, 0, mat));

  // Create second input
  paddle_matrix roi = paddle_matrix_create(
      /* sample_num */ data->roi[0].c, // roi number
      /* size */ 5, // roi dimension
      /* useGPU */ false);
  CHECK_PD(paddle_matrix_set_value(roi, data->roi[0].data));
  CHECK_PD(paddle_arguments_set_value(in_args, 1, roi));

  // Set the frame shape.
  int height = data->input[0].h;
  int width = data->input[0].w;
  CHECK_PD(paddle_arguments_set_frame_shape(in_args, 0, height, width));

  // Step 4: Do inference.
  paddle_arguments out_args = paddle_arguments_create_none();
  CHECK_PD(paddle_gradient_machine_forward(machine,
                                        in_args,
                                        out_args,
                                        /* isTrain */ false));

  // Step 5: Get the result.
  paddle_matrix probs = paddle_matrix_create_none();
  CHECK_PD(paddle_arguments_get_value(out_args, 0, probs));

  CHECK_PD(paddle_matrix_get_shape(probs, result_height, result_width));
  CHECK_PD(paddle_matrix_get_row(probs, 0, &result));

  // Step 6: Release the resources.
  CHECK_PD(paddle_arguments_destroy(in_args));
  CHECK_PD(paddle_matrix_destroy(mat));
  CHECK_PD(paddle_matrix_destroy(roi));
  CHECK_PD(paddle_arguments_destroy(out_args));
  CHECK_PD(paddle_matrix_destroy(probs));
}

void release(paddle_gradient_machine& machine) {
  if (machine != nullptr) {
    CHECK_PD(paddle_gradient_machine_destroy(machine));
  }
}

void verify(float* raw, float* pred, const uint64_t height, const uint64_t width) {
  float eps = 1e-4;
  for (uint64_t i = 0; i < height; ++i) {
    for (uint64_t j = 0; j < width; ++j) {
      float a = raw[i * width + j];
      float b = pred[i * width + j];
      if (std::fabs(a - b) > eps) {
        if ((std::fabs(a - b) / std::fabs(a)) > (eps / 10.0f)) {
          printf("There is diff: i = %d, j = %d, output = %f, predict = %f \n", i, j, a, b);
        }
      }
    }
  }
}

int main() {
  init_paddle();

  printf("load model... \n");
  const char* merged_model_path = "models/ocr_frcnn.paddle";
  paddle_gradient_machine machine = init(merged_model_path);

  printf("read data... \n");
  FrcnnReader reader;
  reader.open("data/data_1.bin");
  FrcnnData* data = reader.next();

  printf("c = %d, h = %d, w = %d, size = %d \n",
         data->input[0].c, data->input[0].h,
         data->input[0].w, data->input[0].size);
  
  float* result = nullptr;
  uint64_t result_height = 0;
  uint64_t result_width = 0;
  infer(machine, data, result, &result_height, &result_width);
  CHECK((result_height * result_width) == data->prob[0].size);
  printf("The size of prediction probability is (%d, %d) \n", result_height, result_width);
  verify(data->prob[0].data, result, result_height, result_width);
  release(machine);
  return 0;
}
