# -*- encoding:utf-8 -*-
from scipy.sparse import csr_matrix
import numpy

'''
网络重建指标
计算embedding结果的precision@k
计算embedding结果的ap&&map
'''

# sim_matrix, adj_matrix 都为稀疏矩阵
# ks 为[k1, k2, k3]
def metric_precision_ks(sim_matrix, adj_matrix, ks):
    n,_         = sim_matrix.shape
    batch_size  = 1000
    start       = 0
    n_k         = len(ks)
    predict     = [0]*n_k
    total       = [0]*n_k
    print(numpy.sum(adj_matrix), adj_matrix.shape)
    while True:
        if start >= n:
            break
        print("Begin To Cal %d"%start)
        end = min(start + batch_size,n)
        batch_sim = sim_matrix[start:end,:].toarray()
        batch_adj = adj_matrix[start:end,:].toarray()
        index = (batch_sim).argsort()
        L2 = tuple([batch_adj[i][index[i]] for i in range(end-start)])
        batch_adj_sort = numpy.vstack(L2)
        for i in range(n_k):
            k = ks[i]
            tmp_batch_adj_sort = batch_adj_sort[:,:k]
            total[i] += (end-start)*k
            predict[i] += numpy.sum(tmp_batch_adj_sort)
        start = end
    print(predict)
    print(total)
    result = [float(i)/j for i,j in zip(predict,total)]
    return result




def metric_map(sim_matrix, adj_matrix):
    n,_ = sim_matrix.shape
    aps = []
    for j in range(n):
        batch_sim = sim_matrix[j:j+1,:].toarray()
        batch_adj = adj_matrix[j:j+1,:].toarray()
        index = (batch_sim).argsort()
        tmp_batch_adj_sort = batch_adj[0,index]
        ks = numpy.argwhere(tmp_batch_adj_sort>1e-10)[1:,1]
        if len(ks) == 0:
            continue
        x = numpy.mean(numpy.arange(1,len(ks)+1)/ks)
        aps.append(x)
        print("Begin to cal %d: %5.4f %6d"%(j,x,numpy.sum(batch_adj)))
    return numpy.mean(aps)


if __name__ == "__main__":
    
    a = [[1,0,1],[0,1,1],[1,1,1]]
    s = [[1,0.1,0.8],[0.1,1,0.05],[0.8,0.05,1]]
    ac = csr_matrix(a)
    sc = csr_matrix(s)
    ks = [2,3]
    ps = metric_precision_ks(sc, ac, ks)
    maps = metric_map(sc, ac)
    print(ps)
    print(maps)



        



    
