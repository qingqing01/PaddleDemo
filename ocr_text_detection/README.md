- Build C-API library of PaddlePaddle on linux

  ```
  DEST_ROOT=/home/dangqingqing/mobile/c-api/install
  PADDLE_ROOT=/home/dangqingqing/Paddle
  THIRD_PARTY_ROOT=/home/dangqingqing/.third_party/
  cmake $PADDLE_ROOT -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_ROOT \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_C_API=ON \
        -DWITH_PYTHON=OFF \
        -DWITH_MKLML=OFF \
        -DWITH_MKLDNN=OFF \
        -DWITH_GPU=OFF \
        -DWITH_SWIG_PY=OFF \
        -DWITH_GOLANG=OFF \
        -DWITH_STYLE_CHECK=OFF \
  make
  make install
  ```

- Prepare Model and Data

  `models/ocr_frcnn.tar.gz` is a fast-rcnn model with PaddlePaddle format for OCR detection. This model should be merged before inferring by PaddlePaddle C-API. You should install PaddlePaddle before using `modesl/merge_model.py` to merge it:

  ```
  cd models
  python merge_model.py
  ```

  `data/data_1.bin` and `data/data_2.bin` are two testing data, which contains the input image, the Region-of-Interests(RoIs) and the prediction results. The prediction results are used to verify the predicted results by PaddlePaddle C-API.

- Build C-API for OCR dection

  ```bash
  export PADDLE_ROOT=/home/dangqingqing/mobile/c-api/install
  mkdir build
  cd build
  cmake ..
  make
  cd ..
  ```

  Run the demo:

  ```
  ./build/ocr_frcnn
  ```


- Directory structure

  ```tree
  ├── frcnn_reader.cpp        Data reader class
  ├── frcnn_reader.h          Data reader class
  ├── main.cpp                The C inference demo
  ├── models
  │   ├── merge_model.py      This file is used to merge model
  │   ├── ocr_det.py          The Python inference demo
  │   ├── ocr_frcnn.paddle    The merged model by `merge_model.py`
  │   └── ocr_frcnn.tar.gz    The fast-rcnn model in PaddlePaddle format
  ```
