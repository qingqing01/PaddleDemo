# Copyright (c) 2017 VisualDL Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================

import numpy as np
import paddle
import paddle.fluid as fluid
from visualdl import LogWriter
from PIL import Image


# define a LeNet-5 nn
def lenet_5(img):
    conv1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv2 = fluid.nets.simple_img_conv_pool(
        input=conv1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")

    predition = fluid.layers.fc(input=conv2, size=10, act="softmax")

    return predition


def test():
    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im
    #place = fluid.CUDAPlace(0)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    model_path = 'mnist_save_model'
    infer_program, feeds, fetches = fluid.io.load_inference_model(model_path, exe,
        model_filename='model', params_filename='params')
    im = load_image('test.png')
    results , = exe.run(infer_program,
                        feed={feeds[0]: im},
                        fetch_list=fetches)
    num = np.argmax(results)
    prob = results[0][num]
    print("Inference result, prob: {}, number {}".format(prob, num))


if __name__ == "__main__":
    test()
