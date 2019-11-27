# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import argparse
import sys, os.path as osp
import mxnet as mx
import tvm
from tvm.contrib import cc
import tvm.relay as relay
import numpy as np

parser = argparse.ArgumentParser(description='Benchmark inference performance for CNN models.')
parser.add_argument('--model', type=str, default='resnet18_v1',
                        choices=['resnet18_v1', 'resnet50_v1', 'resnet101_v1', 'vgg16', 'squeezenet1.0', 
                                 'inceptionv3', 'mobilenet1.0', 'mobilenetv2_1.0'],
                        help='pretrained model')
parser.add_argument('--batch-size', type=int, default=1,
                        help='number of images processed each time')
parser.add_argument('--data-shape', type=str, default=224,
                        help='input data shape')
parser.add_argument('--opt-level', type=int, default=3,
                        help='optimization level when using TVM')

opt = parser.parse_args()
if 'inceptionv3' == opt.model:
    opt.data_shape = 299
print(opt)

######################################################################
# Download pretrained model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from mxnet.gluon.model_zoo.vision import get_model
# from matplotlib import pyplot as plt
block = get_model(opt.model, pretrained=True)
print("Successfully loaded %s model" % opt.model)
bs = opt.batch_size
num_iters = 100
data_shape = opt.data_shape
input_shape = (bs, 3, data_shape, data_shape)
target_dir = './tvm_model'
if not osp.isdir(target_dir):
    os.mkdir(target_dir)

# convert Gluon HybrideBlock to Symbol
def get_symbolic_model(net):
    net.hybridize(static_alloc=True, static_shape=True)
    net(mx.nd.ones(shape=input_shape))
    prefix = target_dir + '/' + opt.model
    net.export(prefix, epoch=0)
    print("Successfully saved mxnet symbolic model into %s" % target_dir)
    sym, args_param, aux_param = mx.model.load_checkpoint(prefix, epoch=0)
    return sym, args_param, aux_param

def inference_symbolic_model(sym, args_param, aux_param, bs, num_iters=100, ctx=[mx.cpu()]):
    mod = mx.module.Module(sym, data_names=('data',), label_names=None, fixed_param_names=sym.list_arguments())
    mod.bind(data_shapes=[('data', input_shape)], for_training=False, grad_req=None)
    mod.set_params(args_param, aux_param)

    size = num_iters * bs
    data = mx.random.uniform(-1.0, 1.0, shape=input_shape, dtype='float32')
    batch = mx.io.DataBatch([data], [])

    import time
    dry_run = 10
    for n in range(dry_run + num_iters):
        if n == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        out = mod.get_outputs()
        out[0].asnumpy()
    toc = time.time()
    speed = size / (toc - tic)
    if bs == 1:
        latency = (toc - tic) / size
        print("Inference on %d batches, latency is %f ms" % (num_iters, latency * 1000))
    else:
        print("Inference on %d batches, batch size %d, throughput is %f img/sec" % (num_iters, bs, speed))

print("--------------------- Benchmarking with MXNet-mkl ---------------------")
sym, args_param, aux_param = get_symbolic_model(block)
inference_symbolic_model(sym, args_param, aux_param, bs=bs, num_iters=num_iters)
# sys.exit()

######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
data = np.random.normal(size=input_shape)
shape_dict = {'data': input_shape}
mod, params = relay.frontend.from_mxnet(block, shape_dict)
## we want a probability so add a softmax operator
func = mod["main"]
func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

# save compliled model
def save_tvm_model(name, graph, lib, params):
    deploy_lib = osp.join(target_dir, name + '.o')
    deploy_so = osp.join(target_dir, name + '.so')
    lib.save(deploy_lib)
    cc.create_shared(deploy_so, [deploy_lib])

    with open(osp.join(target_dir, name + "-tvm.json"), "w") as fo:
        fo.write(graph)

    with open(osp.join(target_dir, name + "-tvm.params"), "wb") as fo:
        fo.write(relay.save_param_dict(params))
    print("Successfullt saved TVM optimized models into %s" % target_dir)


print("--------------------- Benchmarking with TVM ---------------------")
######################################################################
# now compile the graph
target = 'llvm -mcpu=cascadelake'
# target = 'llvm -mcpu=core-avx2'
with relay.build_config(opt_level=opt.opt_level):
    graph, lib, params = relay.build(func, target, params=params)

save_tvm_model(opt.model, graph, lib, params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime

ctx = tvm.cpu(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# execute
# m.run()
ftimer = m.module.time_evaluator("run", ctx, number=3, repeat=num_iters)
# prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
if bs == 1:
    print("Inference on %d batches, latency is %f ms" % (num_iters, ftimer().mean * 1000))
else:
    speed = bs / (ftimer().mean)
    print("Inference on %d batches, batch size %d, throughput is %f img/sec" % (num_iters, bs, speed))

# get outputs
tvm_output = m.get_output(0)

