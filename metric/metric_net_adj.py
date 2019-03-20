# -*- encoding:utf-8 -*-
from scipy.sparse import csr_matrix,lil_matrix
import numpy
from sklearn.metrics import pairwise_distances
'''
根据embedding结果计算相似度矩阵
'''


def metric_net_adj(embedding, euclidean=True):
    n,_ = embedding.shape
    adj = lil_matrix((n,n))
    batch_size = 1000
    start_x = 0
    start_y = 0
    while start_x < n:
        end_x = min(n,start_x+batch_size)
        start_y = 0
        while start_y < n:
            end_y = min(n,start_y+batch_size)
            subx_embedding = embedding[start_x:end_x,:]
            suby_embedding = embedding[start_y:end_y,:]
            if euclidean:
                sub_sim = 1 - pairwise_distances(subx_embedding, suby_embedding, metric="euclidean")
            else:
                sub_sim = 1 - pairwise_distances(subx_embedding, suby_embedding, metric="cosine")
            adj[start_x:end_x,start_y:end_y] = sub_sim
            start_y = end_y
        start_x = end_x
    return csr_matrix(adj)


if __name__ == "__main__":
    embedding = csr_matrix(numpy.random.rand(100,10))
    adj = metric_net_adj(embedding)
    print(adj.shape)
    print(adj[0:10,0:10])
    print(len(adj.nonzero()[0]))
