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

class MixedFxper(ExprFunctor):

    def __init__(self, designated_type, first_input, with_transform = True):
        """
        Pararmeters
        -----------
        designated_dtype : string
            The designated dtype(fxp8_x, fxp16_x)

        first_input : string
            The name of this model's first input

        with_transform : bool
            True if first layer is transform op (transpose, reshape)
            False if first layer is conv2d
            
        """
        super().__init__()
        self.dtype = designated_type
        self.first_input = first_input
        self.fxp_mod = True
        self.pattern_flag = False
        self.with_transform = True

    def visit_function(self, fn):
        """
        A functional visitor over Expr.
        recursively traverses the AST and reconstructs the AST.
        """
        print('visit function with ret_type', fn.ret_type)
        new_body = self.visit(fn.body)
        new_params = [self.visit(x) for x in fn.params]

        # check if the first input of final call is a var && present as original type
        if isinstance(new_body.args[0], relay.Var):
            if new_body.args[0].checked_type.dtype != self.dtype:
                return fn
        else:
            return relay.Function(
                list(new_params),
                new_body,
                TensorType(fn.ret_type.shape, self.dtype),
                fn.type_params,
                fn.attrs)

    def peek(self, call):
        """peek for desired pattern
        The call here is a max_pool2d node, peek deeper to check desired pattern
        Trace flow : maxpool2d --> relu --> bias_add --> conv2d --> (transpose/reshape) --> first_input 
        """
        supposed_relu = call.args[0]
        print('peek : ', supposed_relu.op.name)
        if isinstance(supposed_relu, relay.Call) and supposed_relu.op.name == 'nn.relu':
            supposed_bias_add = supposed_relu.args[0]
            print('peek : ', supposed_bias_add.op.name)
            if isinstance(supposed_bias_add, relay.Call) and supposed_bias_add.op.name == 'nn.bias_add':
                supposed_conv2d = supposed_bias_add.args[0]
                print('peek : ', supposed_conv2d.op.name)
                if isinstance(supposed_conv2d, relay.Call) and supposed_conv2d.op.name == 'nn.conv2d':
                    if self.with_transform == True:
                        transfrom_op = supposed_conv2d.args[0]
                        conv2d_weight = supposed_conv2d.args[1]
                        if (isinstance(transfrom_op, relay.Call) and (transfrom_op.op.name == 'transpose' or transfrom_op.op.name == 'reshape')) and isinstance(conv2d_weight, relay.Var):
                            if transfrom_op.args[0].name_hint == self.first_input:
                                print('catch pattern with input : {}'.format(self.first_input))
                                self.pattern_flag = True
                    else:
                        first_input = supposed_conv2d.args[0]
                        conv2d_weight = supposed_conv2d.args[1]
                        if isinstance(first_input, relay.Var) and isinstance(conv2d_weight, relay.Var):
                            if first_input.name_hint == self.first_input:
                                print('catch pattern with input : {}'.format(self.first_input))
                                self.pattern_flag = True
        else:
            pass

    def visit_call(self, call):
        """Target the pattern (with_transform = True) :
             max_pool2d
                \
               relu
                 \
             bias_add
             /      \
            bias    conv2d
                    /    \
                kernel   transpose/reshape
                             |
                           input
        convert args before relu as float32
        append a Cast to cast the output of relu
        Product : 
              .....
                |
             max_pool2d(fxp)
                \
              cast
               \
              relu(fp32)
                \
             bias_add(fp32)
             /           \
            bias(fp32)    conv2d(fp32)
                          /          \
                    kernel(fp32)   transpose/reshape(fp32)
                                         |
                                       input(fp32)
        """
        print('visit call :', call.op.name)
        # check pattern
        if call.op.name == 'nn.max_pool2d' and self.pattern_flag == False:
            print('start peeking')
            self.peek(call)
        # set fxp mode
        if call.op.name == 'nn.relu' and self.pattern_flag == True:
            self.fxp_mod = False

        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]

        #　append cast into bias_add's arg
        if self.pattern_flag == True and call.op.name == 'nn.max_pool2d':
            appended_args = []
            appended_args.append(relay.cast(new_args[0], dtype=self.dtype))
            self.pattern_flag = False
            print('set pattern_flag to False')
            return relay.Call(new_fn, appended_args, call.attrs)
                
        # end of original dtype unify
        if self.fxp_mod == False and call.op.name == 'nn.relu':
            print('set fxp_mod to True')
            self.fxp_mod = True
        
        return relay.Call(new_fn, new_args, call.attrs)

    def visit_var(self, var):
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
        print('visit global_var')
        return gvar

    def visit_op(self, op):
        print('visit op', op)
        return op

    def visit_constant(self, const):
        print('visit constant')
        return relay.const(const.value, dtype=const.dtype)

    # def visit_if(self, ite):
    #     return relay.If(
    #         self.visit(ite.cond),
    #         self.visit(ite.true_branch),
    #         self.visit(ite.false_branch))

    # def visit_let(self, let):
    #     new_var = self.visit(let.var)
    #     new_val = self.visit(let.value)
    #     new_body = self.visit(let.body)
    #     return relay.Let(new_var, new_val, new_body)

    # def visit_constructor(self, con):
    #     return con

    # def visit_match(self, m):
    #     return relay.Match(
    #         self.visit(m.data),
    #         [Clause(c.lhs, self.visit(c.rhs)) for c in m.clauses],
    #         complete=m.complete)

    # def visit_ref_create(self, r):
    #     return relay.RefCreate(self.visit(r.value))

    # def visit_ref_write(self, r):
    #     return relay.RefWrite(self.visit(r.ref), self.visit(r.value))

    # def visit_ref_read(self, r):
    #     return relay.RefRead(self.visit(r.ref))