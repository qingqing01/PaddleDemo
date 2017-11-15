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

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct DataMeta {
  int h;
  int w;
  int c;
  int size;
  float* data;
};

struct FrcnnData {
  std::vector<DataMeta> input;
  std::vector<DataMeta> roi;
  std::vector<DataMeta> prob;
  std::vector<DataMeta> bbox;
};

class FrcnnReader {
 public:
  FrcnnReader();
  ~FrcnnReader();
  
  void open(const char* file_path);
  void close();
  FrcnnData* next();

 private:
  void clear();
  FILE * fp_;
  FrcnnData data_;
};
