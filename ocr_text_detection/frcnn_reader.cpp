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

#include "frcnn_reader.h"

FrcnnReader::FrcnnReader() : fp_(NULL) {}

FrcnnReader::~FrcnnReader() {
  close();
}

void FrcnnReader::open(const char* file_path) {
  close();
  fp_ = fopen(file_path, "rb");
}

void FrcnnReader::close() {
  if (fp_ != NULL) {
    fclose(fp_);
    fp_ = NULL;
  }
  clear();
}

void FrcnnReader::clear() {
  for (size_t i = 0; i < data_.input.size(); i++) {
    if (data_.input.at(i).data)
      delete[] data_.input.at(i).data;
  }
  data_.input.clear();

  for (size_t i = 0; i < data_.roi.size(); i++) {
    if (data_.roi.at(i).data)
      delete[] data_.roi.at(i).data;
  }
  data_.roi.clear();

  for (size_t i = 0; i < data_.prob.size(); i++) {
    if (data_.prob.at(i).data)
      delete[] data_.prob.at(i).data;
  }
  data_.prob.clear();

  for (size_t i = 0; i < data_.bbox.size(); i++) {
    if (data_.bbox.at(i).data)
      delete[] data_.bbox.at(i).data;
  }
  data_.bbox.clear();
}

void HWC2CHW(float* src, const int C, const int H, const int W, float* dst) {
  for (int i = 0; i < C; ++i) {
    for (int j = 0; j < H; ++j) {
      for (int k = 0; k < W; ++k) {
        dst[i * H * W + j * W + k] = src[j * W * C + k * C + i];
      }
    }
  }
}

void ConvertROI(float* roi, const int num) {
  for (int i = 0; i < num; i += 5) {
    roi[i + 3] += roi[i + 1];
    roi[i + 4] += roi[i + 2];
  }
}

FrcnnData* FrcnnReader::next() {
  if (fp_ == NULL) {
    return NULL;
  }
  clear();
  if (feof(fp_)) {
    close();
    return NULL;
  }

  int in_size = 0;
  fread(&in_size, sizeof(int), 1, fp_);
  if (feof(fp_)) {
    close();
    return NULL;
  }
  for (int i = 0; i < in_size / 2; i++) {
    // read input
    int w = 0;
    int h = 0;
    int c = 0;
    fread(&w, sizeof(int), 1, fp_);
    fread(&h, sizeof(int), 1, fp_);
    fread(&c, sizeof(int), 1, fp_);
    int data_len = c * w * h;
    float* data_buf = new float[data_len];
    fread(data_buf, sizeof(float) * data_len, 1, fp_);

    // switch order
    float* data_chw = new float[data_len];
    HWC2CHW(data_buf, c, h, w, data_chw);

    DataMeta input;
    input.data = data_chw;
    input.w = w;
    input.h = h;
    input.c = c;
    input.size = data_len;
    data_.input.push_back(input);

    // read roi
    fread(&w, sizeof(int), 1, fp_); // roi_dim
    fread(&h, sizeof(int), 1, fp_); // roi_one == 1
    fread(&c, sizeof(int), 1, fp_); // roi_num
    printf("h, w, c = %d, %d, %d \n", h, w, c);
    int roi_len = c * w * h;
    float* roi_buf = new float[data_len];
    fread(roi_buf, sizeof(float) * roi_len, 1, fp_);
    ConvertROI(roi_buf, roi_len);

    DataMeta roi;
    roi.data = roi_buf;
    roi.w = w; // roi_dim
    roi.h = h; // roi_one
    roi.c = c; // roi_num
    roi.size = roi_len;
    data_.roi.push_back(roi);
  }

  int output_size = 0;
  fread(&output_size, sizeof(int), 1, fp_);
  for (int i = 0; i < output_size / 2; i++) {
    int output_len = 0;
    // read bbox
    fread(&output_len, sizeof(int), 1, fp_);
    float* bbox_buf = new float[output_len];
    fread(bbox_buf, sizeof(float) * output_len, 1, fp_);
    DataMeta bbox;
    bbox.data = bbox_buf;
    bbox.size = output_len;
    data_.bbox.push_back(bbox);

    // read prob
    fread(&output_len, sizeof(int), 1, fp_);
    float* prob_buf = new float[output_len];
    fread(prob_buf, sizeof(float) * output_len, 1, fp_);
    DataMeta prob;
    prob.data = prob_buf;
    prob.size = output_len;
    data_.prob.push_back(prob);
  }
  return &data_;
}
