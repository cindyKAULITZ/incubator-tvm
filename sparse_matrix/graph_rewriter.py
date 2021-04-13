import tvm
import tvm.relay as relay
from tvm.relay import ExprFunctor
from tvm.relay import build_module
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

    def show_mod(self):
        print(self.mod)

    def update_fn(self, fn):
        self.mod['main'] = fn
        self.show_mod()

    def search_op(self):
        self.shelper.transform_conv2d()
        compress_weight = self.shelper.visit(self.mod['main'])
        self.update_params(compress_weight)
        print("search finish")
        return compress_weight

    def transform_conv2d(self):
        print("into transform weight")
        self.shelper.transform_conv2d()
        compress_weight = self.shelper.visit(self.mod['main'])
        print("compress weight = ", compress_weight)
        self.update_params(compress_weight)
        # self.draw_params(compress_weight, "weight_distribution")

        print("into transform op")
        self.chelper.transform_conv2d(self.params,self.mod,compress_weight)
        updated_fn = self.chelper.visit(self.mod['main'])
        self.update_fn(updated_fn)
        print("update finish")
        # self.mod["main"] = build_module.bind_params_by_name(self.mod["main"], self.params)
        return self.mod, self.params

    def update_params(self, compress_weight):
        print("--------------update params-----------------")
        for key in compress_weight:
            name = str(key.name)
            kernel = self.params[name].asnumpy()
            FN, C, FH, FW = kernel.shape
            print(name)
            # print(kernel.shape)
            col_kernel = kernel.reshape(FN, -1)
            # col_kernel = kernel.reshape(FN, -1).T
            self.params[name] = tvm.nd.array(col_kernel)
            # print(self.mod['main'].params[1])
            # self.mod['main'].params[1] = tvm.nd.array(col_kernel)
            # print(self.params[name].asnumpy().shape)

    def draw_params(self, compress_weight, save_name):
        
        total = len(compress_weight)
        fig, axs = plt.subplots(6,int((total/6)+1), figsize=(100,100),dpi=140, facecolor='w', edgecolor='k')
    #     fig.subplots_adjust(hspace = .5, wspace=.01)

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
        plt.savefig("im2col_"+save_name+"_pic6.png")

        
        

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

    def transform_conv2d(self,params,mod, compress_weight):
        self.__init_needed_var()
        self.params = params
        self.mod = mod
        self.compress_weight = compress_weight
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

        return relay.Function(
            list(new_params),
            new_body,
            TensorType(fn.ret_type.shape, fn.ret_type.dtype),
            fn.type_params,
            fn.attrs)


    def visit_call(self, call):
        print('visit call :', call.op.name)

        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        new_args_len = len(new_args)
        
        
        if call.op.name == "nn.conv2d":
            to_replace = False
            for key in self.compress_weight:
                if new_args[1].name_hint == str(key.name):
                    to_replace = True
                    break
            
            # print(type(new_args[0]))
            # self.count+=1
            if self.count == 0 and to_replace : # and call.checked_type.shape[1] != 16 and call.checked_type.shape[2] != 32 :
                # print("new_args[1].type_annotation.shape[1] = ", new_args[1].type_annotation.shape[1])
                # if new_args[1].type_annotation.shape[1] < 32:
                # check pattern
                print('catch op : ', call.op.name)
                print('catch op checked_type: ',call.checked_type.shape)
                print('op args name :', new_args[1].name_hint)
                print('op weight shape : ',new_args[1].type_annotation.shape)
                print('op new_args[1].type_annotation : ',new_args[1].type_annotation)
                print('op weight type : ',type(new_args[1]))
                # print('op weight shape : ',type(new_args[1].op))
                
                print('append cast')
                # need to append new_args too.
                appended_args = []

                # appended_args.append(relay.nn.im2col_transform(new_args[1] ,kernel_shape=new_args[1].type_annotation.shape ,strides=call.attrs.strides ,padding=call.attrs.padding ,channels=call.attrs.channels,kernel_size=call.attrs.kernel_size,transform_tag="weight"))
                # appended_args.append(relay.nn.im2col_transform(new_args[0] ,kernel_shape=new_args[1].type_annotation.shape ,strides=call.attrs.strides ,padding=call.attrs.padding ,channels=call.attrs.channels,kernel_size=call.attrs.kernel_size,transform_tag="data"))
                # # dense(weight, data)
                # appended_args.append(relay.nn.dense(appended_args[0],appended_args[1]))

                # return relay.reshape(appended_args[2],call.checked_type.shape)
            
                ###################################################################################
                #                           WEIGHT OFFLINE TRANSFORM                              #
                ###################################################################################
                ####### TODO : how to transform weight offline using better implementation ########
                kernel_shape = new_args[1].type_annotation.shape
                for key in self.compress_weight:
                    if str(key.name) == new_args[1].name_hint:
                        kernel_shape = key.kernel_shape
                appended_args.append(relay.nn.im2col_transform(new_args[0] ,kernel_shape=kernel_shape ,strides=call.attrs.strides ,padding=call.attrs.padding ,channels=call.attrs.channels,kernel_size=call.attrs.kernel_size,transform_tag="data"))
                # print('it shoule be :',kernel.shape[0], ',', new_args[1].type_annotation.shape[1]*call.attrs.kernel_size[0]*call.attrs.kernel_size[1] )
                
                # dense(weight, data)
                # appended_args.append(relay.nn.dense(new_args[1],appended_args[0]))
                appended_args.append(relay.nn.dense(appended_args[0],new_args[1]))
                # appended_args.append(relay.transpose(appended_args[1]))
                
                
                # return relay.reshape(appended_args[2],call.checked_type.shape)
                return relay.reshape(appended_args[1],call.checked_type.shape)
                
        print('end of call : ', call.op.name, ' with output-dtype : ', call.type_args)
        return relay.Call(new_fn, new_args, call.attrs)

    def visit_var(self, var):
        print('visit var : ', var.name_hint, ' with dtype - ', var.checked_type.dtype)
        for key in self.compress_weight:
            if var.name_hint == str(key.name):
                print("hi ",var.name_hint)
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
        return relay.const(const.value, dtype=const.dtype)


class SearchOpWeightHelper(ExprFunctor):

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

    def transform_conv2d(self):
        self.__init_needed_var()
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
        return self.compress_weight


    def visit_call(self, call):
        print('visit call :', call.op.name)

        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        
        if call.op.name == "nn.conv2d":
            # self.count+=1
            if self.count == 0 : # and call.checked_type.shape[1] != 16 and call.checked_type.shape[2] != 32:
                # check pattern
                print('catch op : ', call.op.name)
                # if new_args[1].type_annotation.shape[1] < 32:
                self.compress_weight.append(TransformWeight(new_args[1].name_hint, tuple(new_args[1].type_annotation.shape), call.attrs.strides, call.attrs.padding, call.attrs.channels, call.attrs.kernel_size))

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
        return relay.const(const.value, dtype=const.dtype)
