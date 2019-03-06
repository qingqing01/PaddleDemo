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

    #conv1_bn = fluid.layers.batch_norm(input=conv1)

    conv2 = fluid.nets.simple_img_conv_pool(
        input=conv1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")

    predition = fluid.layers.fc(input=conv2, size=10, act="softmax")

    #cost = fluid.layers.cross_entropy(input=predition, label=label)
    #avg_cost = fluid.layers.mean(cost)
    #acc = fluid.layers.accuracy(input=predition, label=label)
    #return avg_cost, acc
    return predition


def test():
    img = fluid.layers.data(name="img", shape=[1, 28, 28], dtype="float32")
    pred = lenet_5(img)
    test_program = fluid.default_main_program().clone(for_test=True)
    #place = fluid.CUDAPlace(0)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.io.load_persistables(exe, 'mnist_model', main_program=test_program)
    
    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im
    
    im = load_image('test.png')
    results, = exe.run(test_program,
        feed={'img': im},
        fetch_list=[pred])
    num = np.argmax(results)
    prob = results[0][num]
    print("Inference result, prob: {}, number {}".format(prob, num))


if __name__ == "__main__":
    test()
