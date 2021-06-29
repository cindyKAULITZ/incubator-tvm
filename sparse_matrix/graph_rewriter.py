import tvm
import tvm.relay as relay
from tvm.relay import ExprFunctor
from tvm.relay import build_module
from tvm.relay.op.nn.nn import dense
from tvm.topi.util import get_const_tuple 
from tvm.relay.ty import (TypeVar, IncompleteType, TensorType, FuncType,
                 TupleType, TypeRelation, RefType, GlobalTypeVar, TypeCall)
import matplotlib.pyplot as plt 
import numpy as np

"""
Useful doc 
-----------
Type related constructor info : python/tvm/relay/ty.py
"""

class TransformWeight:
    def __init__(self, name, kernel_shape, strides, padding, channels, kernel_size):
        self.name = name
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding
        self.channels = channels
        self.kernel_size = kernel_size

    def print(self):
        print("name: ", self.name)
        print("kernel_shape: ", self.kernel_shape)
        print("strides: ", self.strides)
        print("padding: ", self.padding)
        print("channels: ", self.channels)
        print("kernel_size: ", self.kernel_size)


class InsertTransHelper(ExprFunctor):
    """
    To substitute nn.conv2d() by two layout_transform() 
    and a dense() to make conv2d computed by sparse copmress format. 
    """

    def __init__(self, mod, params):
        super().__init__()
        self.chelper = TransHelper()
        self.shelper = SearchOpWeightHelper()
        self.mod = mod
        self.params = params
        self.sparsity_threshold = 0.1
        self.ret_dict = dict()
        

    def show_mod(self):
        print(self.mod)

    def update_fn(self, fn):
        self.mod['main'] = fn
        self.show_mod()

    def search_op(self):
        print("into transform weight")
        self.shelper.transform_conv2d(self.params, self.sparsity_threshold)
        compress_weight, dense_weight = self.shelper.visit(self.mod['main'])
        print("compress weight = ", compress_weight)
        self.compress_weight = compress_weight
        return compress_weight, dense_weight

    def get_transform_list(self, sparsity_threshold):
        self.sparsity_threshold = sparsity_threshold
        compress_weight, dense_weight = self.search_op()
        self.update_params(compress_weight, dense_weight)
        return self.ret_dict

    def transform_conv2d(self, sparsity_threshold, dense_list):
        # self.sparsity_threshold = sparsity_threshold
        # compress_weight, dense_weight = self.search_op()
        # self.update_params(compress_weight, dense_weight)
        # self.draw_params(compress_weight, "weight_distribution")

        print("into transform op")
        self.chelper.transform_conv2d(self.params, self.compress_weight, dense_list)
        updated_fn = self.chelper.visit(self.mod['main'])
        self.update_fn(updated_fn)
        print("update finish")

        return self.mod, self.params, self.ret_dict

    def update_params(self, compress_weight, dense_weight):
        print("--------------update params-----------------")
        for key in compress_weight:
            name = str(key.name)
            kernel = self.params[name].asnumpy()
            FN, C, FH, FW = kernel.shape
            print(name)
            col_kernel = kernel.reshape(FN, -1)
            self.params[name] = tvm.nd.array(col_kernel)
        # #############################################
        # list contain weights of dense and conv2d
        # use for prediciotn list with dense op
        ##############################################
        for key in dense_weight:
            name = key
            # kernel = self.params[name].asnumpy()
            if "weight" in name: 
                print("dense weight : ",name)
                print(self.params[name].asnumpy().shape)
                self.ret_dict[name] = self.params[name].asnumpy()

    # unused: draw distribution of weights
    def draw_params(self, compress_weight, save_name):
        
        total = len(compress_weight)
        fig, axs = plt.subplots(6,int((total/6)+1), figsize=(100,100),dpi=140, facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace = .5, wspace=.01)
        axs = axs.ravel()
        i = 0
        for key in compress_weight:
            name = str(key.name)
            kernel = self.params[name].asnumpy()
            # print(name)
            # print(kernel.shape)
            axs[i].spy(kernel, aspect='auto',markersize=1)
            sparsity = 1 - (np.count_nonzero(kernel) / kernel.size)
            axs[i].set_title(name+"- shape: "+ str(kernel.shape) +" - sparsity: "+str(sparsity))
            i+=1
            
        plt.show()
        plt.savefig("./PNGs/im2col_"+save_name+"_pic.png")

        
        
# transform conv2d op
class TransHelper(ExprFunctor):

    def __init__(self):
        super().__init__()
        self.__init_needed_var()
        self.compress_weight = []
        
    
    def __init_needed_var(self):
        self.designated_dtype = 'float32'
        # for target pattern
        self.call_list = []
        self.arrived_call = []
        self.targeted = 0

    def transform_conv2d(self, params, compress_weight, dense_list):
        self.__init_needed_var()
        self.params = params
        self.compress_weight = compress_weight
        self.dense_list = dense_list
        self.count = 0
        # print('---- setting ------')
        # print('[TransHelper] get target dtype : ', dtype)
        # print('-------------------')


    def visit_function(self, fn):
        """
        A functional visitor over Expr.
        recursively traverses the AST and reconstructs the AST.
        """
        print('visit function with ret_type', fn.ret_type)
        new_body = self.visit(fn.body)
        new_params = [self.visit(x) for x in fn.params]

        # return fn 
        return  relay.Function(
                list(new_params),
                new_body,
                None,
                # TensorType(fn.ret_type.shape, fn.ret_type.dtype),
                fn.type_params,
                fn.attrs)


    def visit_call(self, call):
        print('visit call :', call.op.name)

        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        new_args_len = len(new_args)
        
        
        if call.op.name == "nn.conv2d":
            # in some model, weights of conv2d are preprocess with another op first.
            if str(type(new_args[1])) == "<class 'tvm.relay.expr.Call'>":
                new_args[1] = new_args[1].args[0]

            to_replace = False
            for key in self.compress_weight:
                if new_args[1].name_hint == str(key.name):
                    to_replace = True
                    break
            
            if to_replace :
                # check pattern
                print('catch op : ', call.op.name)
                print('catch op checked_type: ',call.checked_type.shape)
                print('op args name :', new_args[1].name_hint)
                print('op weight shape : ',new_args[1].type_annotation.shape)
                print('op new_args[1].type_annotation : ',new_args[1].type_annotation)
                print('op weight type : ',type(new_args[1]))
                print('op call.attrs.groups: ',call.attrs.groups)
                
                print('append cast')
                # need to append new_args too.
                appended_args = []
        
                ###################################################################################
                #                           WEIGHT OFFLINE TRANSFORM                              
                ###################################################################################
                ####### TODO : how to transform weight offline using better implementation ########
               
                # handle bitwise conv2d
                if call.attrs.groups > 1:
                    return relay.Call(new_fn, new_args, call.attrs)

                kernel_shape = new_args[1].type_annotation.shape
                for key in self.compress_weight:
                    if str(key.name) == new_args[1].name_hint:
                        kernel_shape = key.kernel_shape
                appended_args.append(relay.nn.im2col_transform(new_args[0] ,kernel_shape=kernel_shape,strides=call.attrs.strides ,padding=call.attrs.padding ,channels=call.attrs.channels,kernel_size=call.attrs.kernel_size,transform_tag="data"))
                appended_args.append(relay.nn.dense(appended_args[0],new_args[1]))
                appended_args.append(relay.transpose(appended_args[1]))
                
                return relay.reshape(appended_args[2],call.checked_type.shape)
                
        print('end of call : ', call.op.name, ' with output-dtype : ', call.type_args)
        return relay.Call(new_fn, new_args, call.attrs)

    def visit_var(self, var):
        print('visit var : ', var.name_hint, ' with dtype - ', var.checked_type.dtype)
        # replace with new shape
        for key in self.compress_weight:
            if var.name_hint == str(key.name):
                print("var name : ", var.name_hint)
                self.count+=1
                print(self.params[var.name_hint].shape)
                return relay.var(var.name_hint,shape=self.params[var.name_hint].shape)
        return var

    def visit_global_id(self, global_var):
        return global_var

    def visit_tuple(self, tup):
        return relay.Tuple([self.visit(field) for field in tup.fields])

    def visit_tuple_getitem(self, op):
        tuple_value = self.visit(op.tuple_value)
        if not tuple_value.same_as(op.tuple_value):
            return relay.TupleGetItem(tuple_value, op.index)
        return op

    def visit_global_var(self, gvar):
        return gvar

    def visit_op(self, op):
        return op

    def visit_constant(self, const):
        return const


# search for conv2d weights
class SearchOpWeightHelper(ExprFunctor):

    def __init__(self):
        super().__init__()
        self.__init_needed_var()
        self.compress_weight = []
        self.dense_weight = []
        
    
    def __init_needed_var(self):
        self.designated_dtype = 'float32'
        # for target pattern
        self.call_list = []
        self.arrived_call = []
        self.targeted = 0

    def transform_conv2d(self, params, sparsity_threshold):
        self.__init_needed_var()
        self.count = 0
        self.params = params
        self.sparsity_threshold = sparsity_threshold
        # print('---- setting ------')
        # print('[TransHelper] get target dtype : ', dtype)
        # print('-------------------')


    def visit_function(self, fn):
        """
        A functional visitor over Expr.
        recursively traverses the AST and reconstructs the AST.
        """
        print('visit function with ret_type', fn.ret_type)
        new_body = self.visit(fn.body)
        new_params = [self.visit(x) for x in fn.params]
        return self.compress_weight, self.dense_weight


    def visit_call(self, call):
        print('visit call :', call.op.name)

        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        # dense op:
        # in pytorch : directly weight
        # in onnx : preprocess in other op first
        if call.op.name == "nn.dense":
            # torch file
            # name = str(new_args[0].name_hint)
            name = "dense_tmp"

            # onnx file
            # name = str(new_args[1].name_hint)
            self.dense_weight.append(name)
        if call.op.name == "nn.conv2d":
            print('catch op : ', call.op.name)
            print('op args # :', len(new_args))
            print('op data shape : ',type(new_args[1]))
            # handle bitwise conv2d
            if str(type(new_args[1])) == "<class 'tvm.relay.expr.Call'>":
                new_args[1] = new_args[1].args[0]
                print("[debug] args 1 name:",new_args[1].name_hint)

            name = str(new_args[1].name_hint)
            if self.count == 0 and ("depth" not in name) : 
                # handle bitwise conv2d
                if call.attrs.groups > 1:
                    return relay.Call(new_fn, new_args, call.attrs)
                # check pattern
                w_np = self.params[name].asnumpy()
                sparsity = 1.0 - (np.count_nonzero(w_np) / w_np.size)
                if sparsity >= self.sparsity_threshold:
                    print('catch op : ', call.op.name)
                    self.compress_weight.append(TransformWeight(new_args[1].name_hint, tuple(new_args[1].type_annotation.shape), call.attrs.strides, call.attrs.padding, call.attrs.channels, call.attrs.kernel_size))
                    self.dense_weight.append(new_args[1].name_hint)
                else:
                    print("no transform : ", new_args[1].name_hint," - sparsity: ", sparsity ," - shape : ", new_args[1].type_annotation.shape)
        print('end of call : ', call.op.name, ' with output-dtype : ', call.type_args)
        return relay.Call(new_fn, new_args, call.attrs)

    def visit_var(self, var):
        print('visit var : ', var.name_hint, ' with dtype - ', var.checked_type.dtype)
        return var

    def visit_global_id(self, global_var):
        return global_var

    def visit_tuple(self, tup):
        return relay.Tuple([self.visit(field) for field in tup.fields])

    def visit_tuple_getitem(self, op):
        tuple_value = self.visit(op.tuple_value)
        if not tuple_value.same_as(op.tuple_value):
            return relay.TupleGetItem(tuple_value, op.index)
        return op

    def visit_global_var(self, gvar):
        return gvar

    def visit_op(self, op):
        return op

    def visit_constant(self, const):
        return const