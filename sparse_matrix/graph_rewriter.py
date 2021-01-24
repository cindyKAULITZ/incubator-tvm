import tvm
import tvm.relay as relay
from tvm.relay import ExprFunctor
from tvm.relay.ty import (TypeVar, IncompleteType, TensorType, FuncType,
                 TupleType, TypeRelation, RefType, GlobalTypeVar, TypeCall)

"""
Useful doc 
-----------
Type related constructor info : python/tvm/relay/ty.py
"""

class InsertTransHelper(ExprFunctor):
    """
    To substitute nn.conv2d() by two layout_transform() 
    and a dense() to make conv2d computed by sparse copmress format. 
    """

    def __init__(self, mod):
        super().__init__()
        self.chelper = TransHelper()
        self.mod = mod

    def show_mod(self):
        print(self.mod)

    def update_fn(self, fn):
        self.mod['main'] = fn
        self.show_mod()

    def transform_conv2d(self):
        self.chelper.transform_conv2d()
        updated_fn = self.chelper.visit(self.mod['main'])
        self.update_fn(updated_fn)

    
        

class TransHelper(ExprFunctor):

    def __init__(self):
        super().__init__()
        self.__init_needed_var()
        
    
    def __init_needed_var(self):
        self.designated_dtype = 'float32'
        # for target pattern
        self.call_list = []
        self.arrived_call = []
        self.targeted = 0

    def transform_conv2d(self):
        self.__init_needed_var()
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

        if call.op.name == "nn.conv2d" :
            # check pattern
            print('catch op : ', call.op.name)
            print("meet conv2d with padding : ", call.attrs.padding)
            print("meet conv2d with pakernel_sizedding : ", call.attrs.kernel_size)
            # print('catch op : ', call.op)
            # print('catch op : ', call)
            # print('catch op attrs : ', type(call.op))
            print('append cast')
            # need to append new_args too.
            appended_args = []
            # appended_args = new_args
            # appended_args.append(relay.reshape(new_args[0],(4,4)))
            # appended_args.append(relay.reshape(new_args[1],(9,-1)))
            appended_args.append(relay.nn.im2col_transform(new_args[0],channels=call.attrs.channels,kernel_size=call.attrs.kernel_size,transform_tag="data"))
            appended_args.append(relay.nn.im2col_transform(new_args[1],channels=call.attrs.channels,kernel_size=call.attrs.kernel_size,transform_tag="weight"))
            
            # appended_args.append(relay.nn.contrib_conv2d_gemm_without_weight_transform(new_args[0],new_args[1]))
            # appended_args.append(relay.nn.dense(appended_args[0], appended_args[1]))
            # appended_args.append(relay.layout_transform(new_args[0], "NCHW", "NCHW"))
            # appended_args.append(relay.layout_transform(new_args[1], "HWIO", "OIHW"))
            # appended_args.append(relay.nn.dense(appended_args[0], appended_args[1]))
            for i in range(2, new_args_len):
                appended_args.append(new_args[i])
            
            # return relay.nn.contrib_conv2d_gemm_without_weight_transform(new_args[0],new_args[1],channels=4,kernel_size=(3,3))
            return relay.nn.dense(appended_args[0], appended_args[1])
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
