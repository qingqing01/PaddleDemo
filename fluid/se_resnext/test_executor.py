#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import os
import numpy as np
import math

import paddle.fluid as fluid
import paddle
import paddle.dataset.mnist as mnist
import paddle.dataset.flowers as flowers
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
from se_resnext import SE_ResNeXt


def cosine_decay(lr, step_each_epoch, epochs):
    global_step = _decay_step_counter()
    with init_on_cpu():
        epoch = fluid.layers.floor(global_step / step_each_epoch)
        decayed_lr = lr * (fluid.layers.cos(epoch * (math.pi / epochs)) + 1) / 2
    return decayed_lr


def lenet(data, label):
    conv1 = fluid.layers.conv2d(data, 32, 5, 1, act=None)
    bn1 = fluid.layers.batch_norm(conv1, act='relu')
    pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
    conv2 = fluid.layers.conv2d(pool1, 50, 5, 1, act=None)
    bn2 = fluid.layers.batch_norm(conv2, act='relu')
    pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)

    fc1 = fluid.layers.fc(pool2, size=500, act='relu')
    fc2 = fluid.layers.fc(fc1, size=10, act='softmax')

    loss = fluid.layers.cross_entropy(input=fc2, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss


class CompareParallelExecutorAndParallelDo(unittest.TestCase):
    def exe(self, train_inputs, test_inputs, seed):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed
        with fluid.program_guard(main, startup):
            data = fluid.layers.data(
                name='image', shape=[3, 224, 224], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            out = SE_ResNeXt(input=data, class_dim=102)
            loss = fluid.layers.cross_entropy(input=out, label=label)
            loss = fluid.layers.mean(loss)

            test_program = main.clone(for_test=True)
            #    learning_rate=cosine_decay(0.01, 1, len(train_inputs)),
            opt = fluid.optimizer.Momentum(
                learning_rate=0.01,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))
            opt.minimize(loss)
            #fluid.memory_optimize(main)

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            var = fluid.global_scope().find_var('conv2d_0.w_0').get_tensor()
            #print('do w ', np.array(var))
            #print('exe main ', main)
            grad_var = fluid.framework.get_var('conv2d_0.w_0@GRAD')

            fetch_list = [loss, grad_var]
            feeder = fluid.DataFeeder(place=place, feed_list=[data, label])

            #fetch_vars = []
            #for k, _ in main.blocks[0].vars.iteritems():
            #    fetch_vars.append(k)

            losses = []
            grads = []
            test_losses = []
            for data in train_inputs:
                all_vars = main.global_block().vars
                #all_parameters = {k : v for k, v in all_vars.iteritems() if v.persistable}
                import collections
                all_parameters = collections.OrderedDict()
                for k, v in all_vars.iteritems():
                    if v.persistable and 'velocity' not in k:
                        all_parameters[k] = v
                print('Total vars: %d\n' %(len(all_parameters)))
                for k,v in all_parameters.iteritems():
                    var = fluid.global_scope().find_var(k).get_tensor()
                    print('!!%s: %f\n'%(k, np.sum(np.abs(np.array(var)))))

                ret = exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
                loss_v = np.array(ret[0])
                losses.append(loss_v[0])
                grads.append(np.array(ret[1]))

                #sys.stderr.write('total vars: %d, returned: %d\n' %
                #                 (len(fetch_vars), len(ret)))
                #for i in xrange(2, len(ret)):
                #    sys.stderr.write('!!%s: %s\n' % (fetch_vars[i - 1],
                #                                     np.sum(np.abs(ret[i]))))

                for test_data in test_inputs:
                    test_loss = exe.run(test_program,
                                        feed=feeder.feed(test_data),
                                        fetch_list=[loss])
                    test_losses.append(test_loss[0][0])
            return losses, grads, test_losses


    def test_compare_grad(self):
        #trn_reader = paddle.batch(flowers.test(use_xmap=False), batch_size=16)
        #trn_reader_iter = trn_reader()
        tst_reader = paddle.batch(flowers.test(use_xmap=False), batch_size=16)
        tst_reader_iter = tst_reader()

        seed = 100
        iters = 4
        train_inputs = []
        for i in range(iters):
            train_inputs.append(tst_reader_iter.next())
        test_inputs = [tst_reader_iter.next()]

        exe_losses, exe_grads, exe_test_losses = self.exe(
            train_inputs, test_inputs, seed)

        print 'train loss ', exe_losses
        print 'test loss ', exe_test_losses
        #print 'exe ', exe_grads
        np.save('exe_grads.npy', exe_grads)


if __name__ == '__main__':
    unittest.main()
