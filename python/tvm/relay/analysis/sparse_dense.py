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


SparseAnalysisResult = namedtuple(
    "SparseAnalysisResult",
    [
        "weight_name",
        "weight_shape",
    ],
)


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


def process_params(expr, params, block_size, sparsity_threshold, mode):
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
    memo = SparseAnalysisResult(weight_name=[], weight_shape=[])
    weight_names = _search_dense_op_weight(expr)
    print("in process params")
    print(weight_names)
    # print(weight_names.size)
    print(len(mode))
    print(mode)
    mode = mode[::-1]
    for name, mo in zip(weight_names, mode):
        name = str(name)
        w_np = params[name].asnumpy()
        sparsity = 1.0 - (np.count_nonzero(w_np) / w_np.size)
        print("weight name : ", name)
        print("sparsity : ", sparsity)
        # if sparsity >= sparsity_threshold:
        if sparsity >= sparsity_threshold and ("depth" not in name):
        # if sparsity >= sparsity_threshold and ("conv" in name) and ("depth" not in name):
            sparse_weight = sp.csr_matrix(w_np)
            if mo == 1:
                print("mode bsr")
                print(w_np.size)
                print(block_size[0])
                if w_np.size/block_size[0] == 0:
                    sparse_weight = sp.bsr_matrix(w_np, blocksize=block_size)
                else:
                    sparse_weight = sp.bsr_matrix(w_np, blocksize=(1,1))
            elif mo == 0:
                print("mode csr")
                sparse_weight = sp.csr_matrix(w_np)
            else:
                print("mode nor")
                continue
                assert("Not Implement yet.")
            # sparse_weight = sp.coo_matrix(w_np)

            # print('data : ', sparse_weight.data.shape)
            # print('indiecs : ', sparse_weight.indices.shape)
            # print('indptr : ', sparse_weight.indptr.shape)

            # remove dense weight
            del params[name]
            memo.weight_name.append(name)
            memo.weight_shape.append(
                list(sparse_weight.data.shape)
                + list(sparse_weight.indices.shape)
                + list(sparse_weight.indptr.shape)
            )
            params[name + ".data"] = tvm.nd.array(sparse_weight.data)
            params[name + ".indices"] = tvm.nd.array(sparse_weight.indices)
            params[name + ".indptr"] = tvm.nd.array(sparse_weight.indptr)
    ret = SparseAnalysisResult(
        weight_name=tvm.runtime.convert(memo.weight_name),
        weight_shape=tvm.runtime.convert(memo.weight_shape),
    )
    return ret
