# -*- encoding:utf-8 -*-
import copy
import os
import numpy
import random
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy import io
from .graph import Graph
from sklearn.cluster import KMeans



def subgraph_index(graph, split_num = 2):
    index = []
    for i in range(split_num):
        index.append(set())
    edges = graph.edges
    adj = graph.adjacent_matrix
    edge_number = len(edges)
    node_number = graph.node_number
    weights_matrix = dok_matrix((edge_number, node_number), dtype=int)
    for k in range(edge_number):
        i,j = edges[k]
        weights_matrix[k] = adj[i] + adj[j]
    model = KMeans(split_num, n_jobs = 4)
    classes = numpy.argmax(model.fit_transform(weights_matrix), 1)
    print(
        len(classes),":", 
        len(numpy.argwhere(classes==0)), 
        len(numpy.argwhere(classes==1)), 
        len(numpy.argwhere(classes==2)),
        len(numpy.argwhere(classes==3)) )
    for k in range(edge_number):
        c = classes[k]
        i,j = edges[k]
        index[c].add(i)
        index[c].add(j)
    index = [list(i) for i in index]
    return index

