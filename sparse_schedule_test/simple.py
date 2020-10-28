import tvm
import numpy
import timeit
from topi.nn.pad import pad

# declare needed tensor expr
n = 6
A = tvm.placeholder((1, 1, n, n), name='Input')
B = tvm.placeholder((1, 1, 3, 3), name='Filter')
# temp = pad(A, n, 6, name="pad_temp")

rc = tvm.reduce_axis((0, 1), name='rc')
ry = tvm.reduce_axis((0, 3), name='ry')
rx = tvm.reduce_axis((0, 3), name='rx')

def _mysum(nn,ff,yy,xx):
    a = 0
    b = 0
    for ry in range(0,3):
        for rx in range(0,3):
            tmp = tvm.if_then_else(a >= b, b, a)
            # tmp1 = tvm.if_then_else(tmp == 1, a ,b )
            tmp += A[nn,0, yy+ry, xx+rx]*B[ff,0,ry,rx]
            # if tmp1 == 1:
            #     a = tmp1
            # else:
            #     b = tmp1
            a += tmp
            # a = tvm.if_then_else(tmp2 == a, tmp1, a)
            # b = tvm.if_then_else(tmp2 == b, tmp1, b)
            # a += A[nn,0, yy+ry, xx+rx]*B[ff,0,ry,rx]
    # for i in range(0, len(b)-1):
    #     a+=b[i]
    return a+b


# C = tvm.compute(
#             (1, 1, 4, 4),
#             lambda nn, ff, yy, xx: tvm.sum(
#             A[nn, rc, yy * 1 + ry * 1,
#                  xx * 1 + rx * 1]*
#             B[ff, rc, ry, rx],
#             axis=[rc, ry, rx]), tag="conv2d_nchw")
C = tvm.compute((1,1,4,4),
            lambda nn, ff, yy, xx: 
            _mysum(nn,ff,yy,xx), tag="conv2d_nchw")

s = tvm.create_schedule(C.op)

op_list = [i for i in C.op.input_tensors]
op_list.append(C)

print('op input list : ', op_list)
print('op axis : ', C.op.axis)
print("\n")
print(tvm.lower(s, op_list, simple_mode=True))

# s_state = tvm.placeholder((1,1,6,6),name="state")
# D = tvm.compute((1,1,3,3),
#             lambda t: tvm.sum(
#             s_state[t,rc,ry,rx],
#             axis=[rc, ry, rx]), tag="conv2d_nchw")
# s1 = tvm.create_schedule(D.op)

# op_list1 = [i for i in D.op.input_tensors]
# op_list1.append(D)

# print('op input list1 : ', op_list1)

# s_scan = tvm.scan(C,D,s_state)

# s = tvm.create_schedule(s_scan.op)

# op_list = [i for i in s_scan.op.input_tensors]
# op_list.append(s_scan)

# print('op input list : ', op_list)
# # print('op axis : ', s_scan.op.axis)
# print("\n")
# print(tvm.lower(s, op_list, simple_mode=True))
# print(A)
# # print(temp)
# print(B)
# print(C)

# print(C.input_tensors)

# ----

# f = s[C].fuse()

# print('after fuse')
# print(tvm.lower(s, [A, B, C], simple_mode=True))

# # ----

# fc, fy, fx = s[C].split(B, factor=3)
# print('after split')
# print(tvm.lower(s, [A, B, C], simple_mode=True))

# # --- 
# s[C].vectorize(fi)
# print(tvm.lower(s, [A, B, C], simple_mode=True))

# # ---
# # print(a)
# # a = tvm.ir_pass.LoopPartition(a, True)
# # a = tvm.ir_pass.Simplify(a)
# # a = tvm.ir_pass.VectorizeLoop(a)

# # print(a)
