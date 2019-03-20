# -*- encoding:utf-8 -*-
import copy
import os
import numpy
import random

# 空间融合
# 将new空间转换到old空间，依据new和old之间公共的index做线性变换
# common_embedding_new * V  = commen_embedding_old
# embedding_new * V         = embedding_old
def space_transpose(d_new, embedding_new, d_old, embedding_old):
    common_index_origin     = list( set(d_new.keys()).intersection( set(d_old.keys()) ) )
    common_index_new        = [ d_new[i] for i in common_index_origin]
    common_index_old        = [ d_old[i] for i in common_index_origin]
    print(len(common_index_origin))

    common_embedding_new    = embedding_new[common_index_new]
    common_embedding_old    = embedding_old[common_index_old]
    V = numpy.matmul(numpy.linalg.pinv(common_embedding_new),common_embedding_old)
    print(numpy.sum(numpy.abs(common_embedding_old-numpy.matmul(common_embedding_new,V))))
    embedding_new_transfrom   = numpy.matmul(embedding_new, V)
    return embedding_new_transfrom

if __name__ == "__main__":
    A = numpy.random.rand(5000,100)
    B = numpy.random.rand(5000,100)
    dA,dB = {},{}
    for i in range(5000):
        dA[i]   = i
        if i <= 4000:
            dB[i]   = i
        else:
            dB[i+10000]   = i
    # print(dA, dB)
    BB = space_transpose(dB, B, dA, A)
    # print(A)
    # print(B)
    # print(BB)
    print(numpy.sum(numpy.abs(A-B)))
    print(numpy.sum(numpy.abs(A-BB)))