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
# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
This file contains helper functions for convert dense model
to block sparse model
"""
from collections import namedtuple
import numpy as np
import scipy.sparse as sp
import tvm
from . import _ffi_api
# hhliao

SparseAnalysisResult = namedtuple("SparseAnalysisResult", [
    "weight_name",
    "weight_shape",
])

def _search_dense_op_weight(expr):
    """Search name of weight in all ```nn.dense``` operator
       This is a helpful function to determine which param need
       to be converted to sparse

    Parameters
    ----------
    expr : relay.Expr
        Expr will be searched

    Returns
    -------
    ret : Array[String]
        name of weight in all ``nn.dense``` operator
    """
    return _ffi_api.search_dense_op_weight(expr)


def process_params(expr, params, block_size, sparsity_threshold):
    """[summary]

    Parameters
    ----------
    expr : Relay.Expr
        Expr of the network
    params : Dict[String, tvm.nd.array]
        parameters of the network
    block_size : Tuple(int, int)
        Blocksize in BSR matrix
    sparsity_threshold : float
        Minimal sparsity requirement for converting to sparse operation

    Returns
    -------
    ret : Namedtuple[weight_name: Array[String], weight_shape: Array[Array[IntImm]]]
        return names of qualified dense weight and the shape in BSR format
    """
    print("Enter to process params")
    memo = SparseAnalysisResult(weight_name=[], weight_shape=[])
    weight_names = _search_dense_op_weight(expr)
    # weight_names = params.keys()
    for name in weight_names:
        name = str(name)
        w_np = params[name].asnumpy()
        sparsity = 1.0 - (np.count_nonzero(w_np) / w_np.size)
        print("sparsity = ", sparsity)
        print("name =", name)
        if sparsity >= sparsity_threshold:
            # print("name {} > sparsity threshold"%name)
            # revise from dim = 1 (dense's weight shape:(none,3)) to dim = 4 (conv2d's weight shape:(1,1,3,3))
            # sparse_weight = sp.bsr_matrix(w_np, blocksize=block_size)
            # # remove dense weight
            del params[name]
            memo.weight_name.append(name)
            memo.weight_shape.append((1,5))
            params[name + ".data"] = tvm.nd.array(0)
            params[name + ".indices"] = tvm.nd.array(0)
            params[name + ".indptr"] = tvm.nd.array(0)
    for key in params.keys():
        arr = params[key].asnumpy()
        print("Sparse_dense key's arr : " ,key)
        print(" shape : " , arr.shape)
        print(" value : " , arr)
    ret = SparseAnalysisResult(
        weight_name=tvm.runtime.convert(memo.weight_name),
        weight_shape=tvm.runtime.convert(memo.weight_shape)
    )
    return ret


