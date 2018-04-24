## Build Fluid Shared library
Commands:
```bash
PADDLE_ROOT=/path/of/fluidapi
git clone https://github.com/PaddlePaddle/Paddle.git
git checkout -b luotao1-fluid_infer develop
git pull https://github.com/luotao1/Paddle.git fluid_infer
cd Paddle
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PADDLE_ROOT \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_SWIG_PY=OFF \
      -DWITH_GOLANG=OFF \
      -DWITH_PYTHON=OFF \
      -DWITH_MKL=OFF \ # or ON
      ..
make
make inference_lib_dist
```
## Inference Example Project
- Build:

```bash
PADDLE_ROOT=/path/of/fluidapi
# set environment
CUDA_LIB=/path/of/cuda_library
CUDNN_LIB=/path/of/cudnn_library
# if using mkl library
MKL_LIB=/PADDLE_ROOT/install/mklml/lib 
export LD_LIBRARY_PATH=CUDA_LIB:CUDNN_LIB:MKL_LIB

git clone https://github.com/luotao1/fluid_inference_example.git
cd example
mkdir build
cd build
# using shared library
cmake -DPADDLE_ROOT=PADDLE_ROOT -DLIB_TYPE=shared ..
# or using static library and openblas math library
cmake -DPADDLE_ROOT=PADDLE_ROOT -DLIB_TYPE=static -DMATH_LIB=openblas -DCUDA_LIB=CUDA_LIB ..
# or using static library and mkl math library
cmake -DPADDLE_ROOT=PADDLE_ROOT -DLIB_TYPE=static -DMATH_LIB=mkl -DCUDA_LIB=CUDA_LIB ..
make
```
- Inference:

```bash
cd build
../run.sh
```

- Results:

```txt
FLAGS_dirname: ../model/
FLAGS_feed_var_names: x
FLAGS_fetch_var_names: fc_2.tmp_2
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0117 18:10:00.795655  2572 inference.cc:42] loading model from ../model//__model__.dat
I0117 18:10:00.795984  2572 inference.cc:48] program_desc_str's size: 2787
I0117 18:10:00.796463  2572 inference.cc:88] parameter's name: fc_1.b_0
I0117 18:10:00.796591  2572 inference.cc:88] parameter's name: fc_2.w_0
I0117 18:10:00.796669  2572 inference.cc:88] parameter's name: fc_2.b_0
I0117 18:10:00.796769  2572 inference.cc:88] parameter's name: fc_0.b_0
I0117 18:10:00.796852  2572 inference.cc:88] parameter's name: fc_0.w_0
I0117 18:10:00.796929  2572 inference.cc:88] parameter's name: fc_1.w_0
I0117 18:10:00.797011  2572 inference.cc:122] feed var's name: x
I0117 18:10:00.797086  2572 inference.cc:149] fetch var's name: fc_2.tmp_2
W0117 18:10:04.852438  2572 init.cc:56] 'GPU' is not supported, Please re-compile with WITH_GPU option
dims_i: 1 10
result: 0.135148 0.0850182 0.0942107 0.0784854 0.120748 0.119009 0.123512 0.108898 0.0823114 0.0526599
```
