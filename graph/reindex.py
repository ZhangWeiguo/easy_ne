# -*- encoding:utf-8 -*-

import numpy
import random
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
import networkx as nx
# 近似求解图的哈密顿问题


def bfs(adj_matrix, label, depth_limit=10):
    N,_ = adj_matrix.shape
    G = nx.Graph()
    S = adj_matrix.nonzero()
    for i in range(N):
        G.add_node(i)
    for i,j in zip(S[0],S[1]):
        G.add_edge(i,j)
    L = []
    while len(L) < N:
        source = random.choice(list(set(range(N))-set(L)))
        T = nx.bfs_tree(G, source=source)
        for i in T.nodes():
            if i not in L:
                L.append(i)
    print("L Cal Finish")
    print(len(L), N)
    label_copy = numpy.zeros(N) -1
    adj_matrix_copy = dok_matrix((N, N))
    for k,i in enumerate(L):
        label_copy[k] = label[i]
    print("Label Cal Finish")
    LD = {}
    for k,i in enumerate(L):
        LD[i] = k
    for i in range(N):
        S = adj_matrix[i,:].nonzero()[1]
        ii = LD[i]
        for j in S:
            jj = LD[j]
            adj_matrix_copy[ii,jj] = 1
            adj_matrix_copy[jj,ii] = 1
    adj_matrix_copy.tocsr()
    print("Adj Cal Finish")
    print(numpy.sum(adj_matrix),numpy.sum(adj_matrix_copy))
    return adj_matrix_copy,label_copy



def hamiton(adj_matrix, label, neighbor=5, max_step=5):
    N,_ = adj_matrix.shape
    k = 0
    L = [k]
    while len(L)<N:
        k0 = L[-1]
        k2 = 0
        S = adj_matrix[k0,:].nonzero()[1]
        for i in S:
            if i not in L:
                L.append(i)
                k2 += 1
            if k2 >= neighbor:
                break
        if k2 == 0:
            k0 = random.choice(list(set(range(N))-set(L)))
            L.append(k0)
    print("L Cal Finish")
    label_copy = numpy.zeros(N) -1
    adj_matrix_copy = dok_matrix((N, N))
    for k,i in enumerate(L):
        label_copy[k] = label[i]
    print("Label Cal Finish")
    LD = {}
    for k,i in enumerate(L):
        LD[i] = k
    for i in range(N):
        S = adj_matrix[i,:].nonzero()[1]
        ii = LD[i]
        for j in S:
            jj = LD[j]
            adj_matrix_copy[ii,jj] = 1
            adj_matrix_copy[jj,ii] = 1
    adj_matrix_copy.tocsr()
    print("Adj Cal Finish")
    # print(numpy.sum(adj_matrix),numpy.sum(adj_matrix_copy))
    return adj_matrix_copy,label_copy


if __name__ == "__main__":
    adj_matrix = csr_matrix(numpy.random.randint(0,2,(10,10)))
    label = numpy.random.randint(0,10,(10))
    for i in range(10):
        adj_matrix[i,i] = 1
        for j in range(i+1,10):
            adj_matrix[j,i] = adj_matrix[i,j] 

    adj_matrix_copy,label_copy = bfs(adj_matrix, label)
    print(adj_matrix.todense())
    print(adj_matrix_copy.todense())
    print(label)
    print(label_copy)







