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


# define a LeNet-5 nn
def lenet_5(img, label):
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

    cost = fluid.layers.cross_entropy(input=predition, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predition, label=label)

    return avg_cost, acc, predition


def test():
    img = fluid.layers.data(name="img", shape=[1, 28, 28], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    avg_cost, acc, predition = lenet_5(img, label)
    test_program = fluid.default_main_program().clone(for_test=True)
    
    place = fluid.CPUPlace()
    #place = fluid.CUDAPlace(0)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe = fluid.Executor(place)
    # init all param
    exe.run(fluid.default_startup_program())
    fluid.io.load_persistables(exe, 'mnist_model', main_program=test_program)
    
    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=64)
    test_acc, test_cost = [], []
    for data in test_reader():
        res_cost, res_acc = exe.run(test_program,
            feed=feeder.feed(data),
            fetch_list=[avg_cost.name, acc.name])
        test_cost.append(res_cost)
        test_acc.append(res_acc)
    mloss = np.mean(np.array(test_cost))
    macc = np.mean(np.array(test_acc))
    print("Test loss:{}, acc{}".format(mloss, macc))


if __name__ == "__main__":
    test()
