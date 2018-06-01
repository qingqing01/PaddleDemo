/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <thread>  // NOLINT
#include "gflags/gflags.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(dirname, "", "Directory of the inference model.");
DEFINE_int32(num_threads, 1, "Threads number.");
DEFINE_int32(num_samples, 4, "Total threads number.");
DEFINE_int32(print_outputs, 0, "Whether to print the outputs.");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(use_gpu, 1, "whether use gpu.");
DECLARE_double(fraction_of_gpu_memory_to_use);

void RunInference(
    const std::unique_ptr<paddle::framework::ProgramDesc>& inference_program,
    paddle::framework::Executor* executor, paddle::framework::Scope* scope,
    const std::vector<paddle::framework::LoDTensor*>& cpu_feeds,
    const std::vector<paddle::framework::LoDTensor*>& cpu_fetchs,
    const int repeat_num, const int thread_id = 0)  {
  auto copy_program = std::unique_ptr<paddle::framework::ProgramDesc>(
      new paddle::framework::ProgramDesc(*inference_program));

  std::string feed_holder_name = "feed_" + paddle::string::to_string(thread_id);
  std::string fetch_holder_name =
      "fetch_" + paddle::string::to_string(thread_id);
  copy_program->SetFeedHolderName(feed_holder_name);
  copy_program->SetFetchHolderName(fetch_holder_name);

  // 3. Get the feed_target_names and fetch_target_names
  const std::vector<std::string>& feed_target_names =
      copy_program->GetFeedTargetNames();
  const std::vector<std::string>& fetch_target_names =
      copy_program->GetFetchTargetNames();

  // 4. Prepare inputs: set up maps for feed targets
  std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
  for (size_t i = 0; i < feed_target_names.size(); ++i) {
    feed_targets[feed_target_names[i]] = cpu_feeds[i];
  }

  // 5. Define Tensor to get the outputs: set up maps for fetch targets
  std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
  for (size_t i = 0; i < fetch_target_names.size(); ++i) {
    fetch_targets[fetch_target_names[i]] = cpu_fetchs[i];
  }

  // 6. Run the inference program
  auto &lscope = scope->NewScope();
  executor->CreateVariables(*copy_program, &lscope, 0);
  auto ctx = executor->Prepare(*copy_program, 0);

  std::cout << "Thread " << thread_id << " run " << repeat_num << " times" << std::endl;
  for (int i = 0; i < repeat_num; ++i) {
    // executor->Run(*copy_program, scope, &feed_targets, &fetch_targets, true,
    //               true, feed_holder_name, fetch_holder_name);
    executor->RunPreparedContext(ctx.get(), &lscope, &feed_targets, &fetch_targets, false,
                  false, feed_holder_name, fetch_holder_name);
  }
}


inline uint64_t PosixInNsec() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
}


int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_dirname.empty()) {
    // Example:
    //   ./example --dirname=recognize_digits_mlp.inference.model
    std::cout << "Usage: ./example --dirname=path/to/your/model" << std::endl;
    exit(1);
  }

  // 1. Define place, executor, scope
  // auto place = paddle::platform::CUDAPlace(0);
  paddle::platform::Place place;
  if(FLAGS_use_gpu) {
    place = paddle::platform::CUDAPlace(0);
  } else {
    place = paddle::platform::CPUPlace();
  }
  std::vector<std::string> argvs;
  argvs.push_back("");
  argvs.push_back("--fraction_of_gpu_memory_to_use=0.8");
  paddle::framework::InitGflags(argvs);
  paddle::framework::InitDevices(false);
  auto* executor = new paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  std::string dirname = FLAGS_dirname;

  // 2. Initialize the inference program
  std::string prog_filename = "__model_combined__";
  std::string param_filename = "__params_combined__";
  auto inference_program = paddle::inference::Load(
      executor, scope, dirname + "/" + prog_filename,
      dirname + "/" + param_filename);

  // Generate input
  paddle::framework::LoDTensor input;
  srand(0);
  int num_threads = FLAGS_num_threads;
  std::vector<std::vector<paddle::framework::LoDTensor*>> cpu_feeds;
  cpu_feeds.resize(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    auto* input = new paddle::framework::LoDTensor();
    int64_t batch_size = FLAGS_batch_size;
    float* input_ptr =
        input->mutable_data<float>({batch_size, 1, 28, 28},
                                    paddle::platform::CPUPlace());
    for (int j = 0; j < batch_size * 1 * 28 * 28; ++j) {
      input_ptr[j] = rand() / (static_cast<float>(RAND_MAX));
    }
    cpu_feeds[i].push_back(input);
  }

  std::vector<std::vector<paddle::framework::LoDTensor*>> cpu_fetchs;
  cpu_fetchs.resize(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    auto* output = new paddle::framework::LoDTensor();
    cpu_fetchs[i].push_back(output);
  }

  // RunInference(inference_program, executor, scope,
  //     cpu_feeds[0], cpu_fetchs[0], 5, 0);

  // 3. Optional: perform optimization on the inference_program
  // paddle::platform::EnableProfiler(paddle::platform::ProfilerState::kAll);
  int repeate_num = FLAGS_num_samples / num_threads / 2;
  uint64_t start_ns = PosixInNsec();
  if (num_threads == 1) {
    paddle::framework::InitDevices(false);
    RunInference(inference_program, executor, scope,
        cpu_feeds[0], cpu_fetchs[0], repeate_num, 0);
  } else {
    std::vector<std::thread*> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.push_back(new std::thread([&, i](){
        uint64_t start = PosixInNsec();
        paddle::framework::InitDevices(false);
        auto* exe = new paddle::framework::Executor(place);
        RunInference(inference_program, exe, scope,
            cpu_feeds[i], cpu_fetchs[i], repeate_num, i);
        uint64_t end = PosixInNsec();
        std::cout << "Thread " << i << " Time :" <<(double)(end - start)/1000.0 << std::endl;
      }));
    }
    for (int i = 0; i < num_threads; ++i) {
      threads[i]->join();
      delete threads[i];
    }
  }

  uint64_t end_ns = PosixInNsec();
  std::cout << "Total Time : " <<(double)(end_ns - start_ns)/1000.0 << std::endl;
  // paddle::platform::DisableProfiler(
  //     paddle::platform::EventSortingKey::kDefault,
  //     paddle::string::Sprintf("/tmp/profile_infer"));

  // Get outputs
  if (FLAGS_print_outputs) {
    for (size_t n = 0; n < cpu_fetchs.size(); ++n) {
      for (size_t i = 0; i < cpu_fetchs[n].size(); ++i) {
        auto shape = cpu_fetchs[n][i]->dims();
        std::cout << "shape:" << shape << std::endl;
        std::cout << "result:" << std::endl;
        float* output_ptr = cpu_fetchs[n][i]->data<float>();
        int dim = paddle::framework::product(shape) / shape[0];
        for (int j = 0; j < shape[0]; ++j) {
          int dim = paddle::framework::product(shape) / shape[0];
          std::cout << j << "-th sample: ";
          for (int k = 0; k < dim; ++k) {
            std::cout << " " << output_ptr[j * dim + k];
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
    }
  }

  delete scope;
  delete executor;

  return 0;
}
