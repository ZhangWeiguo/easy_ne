# -*- encoding:utf-8 -*-
import copy
import os
import numpy
import random
import shelve
import time
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy import io
from matplotlib import pyplot
import json
from sklearn.neighbors import kneighbors_graph
'''
Grah成员
self.data                   节点向量，默认是邻接矩阵的行向量
self.label                  节点的标签，数字存储
self.label_dict             标签与实体对应关系，可对应字符串
self.adjacement_matrix      邻接矩阵
self.other_matrix           其他自定义距离矩阵，字典存储
self.node_number
self.edge_number
self.is_weighted
self.is_directed
self.logger

Graph方法
__init__                                初始化
init_from_file                          从CSV导入
save_to_mat                             保存到mat
init_from_mat                           从mat导入
sample                                  采样
    __sample_without_subgraph           随机采样点，返回indexes
    __sample_with_subgraph              子图随机采样，返回indexes



辅助函数
file_to_graph
sdne_sample
line_sample
'''

def cne_sample(graph, batch_size):
    for ids1,ids2,weights in graph.sample_pairwise(batch_size):
        xa = graph.adjacent_matrix[ids1,:].todense()
        xb = graph.adjacent_matrix[ids2,:].todense()
        if len(xa) != batch_size:
            break
        yield xa,xb,weights,ids1,ids2

def sdne_sample(graph, batch_size):
    for ids1,ids2,weights in graph.sample_pairwise(batch_size):
        xa = graph.adjacent_matrix[ids1,:].todense()
        xb = graph.adjacent_matrix[ids2,:].todense()
        yield xa,xb,weights,ids1,ids2

def ae_sample(graph, batch_size):
    all_index = list(range(graph.node_number))
    numpy.random.shuffle(all_index)
    start = 0
    end = graph.node_number
    while start < graph.node_number:
        end = start+batch_size
        index = list(range(start, min(end, graph.node_number)))
        start = end
        xa = graph.adjacent_matrix[index,:].todense()
        yield xa,index


def aene_sample(graph, batch_size):
    for ids1,ids2,weights in graph.sample_pairwise(batch_size):
        x1a = graph.adjacent_matrix[ids1,:].todense()
        x2a = graph.data[ids1,:].toarray()
        x1b = graph.adjacent_matrix[ids2,:].todense()
        x2b = graph.data[ids2,:].toarray()
        yield x1a, x2a, x1b, x2b, weights,ids1,ids2

def line_sample(graph, batch_size):
    for ids1,ids2,weights in graph.sample_pairwise(batch_size):
        yield ids1,ids2,weights

def subsdne_sample(graph, batch_size, nodes):
    for ids1,ids2,weights in graph.sample_pairwise(batch_size):
        new_ids1,new_ids2 = [],[]
        new_weights = []
        for k,(i1,i2) in enumerate(zip(ids1, ids2)):
            if i1 not in nodes and i2 not in nodes:
                continue
            new_ids1.append(i1)
            new_ids2.append(i2)
            new_weights.append(weights[k])
        xa = graph.adjacent_matrix[new_ids1,:][:,nodes].todense()
        xb = graph.adjacent_matrix[new_ids2,:][:,nodes].todense()
        yield xa,xb,new_weights,new_ids1,new_ids2



class deepwalk_sample(object):
        def __init__(self, graph, walk_length):
            self.graph          = graph
            self.walk_length    = walk_length
            self.nodes          = [i for i in range(self.graph.node_number)]
            self.curr_index     = 0
            self.max_index      = self.graph.node_number
            random.shuffle(self.nodes)
        def __iter__(self):
            return self
        def __next__(self):
            if self.curr_index >= self.max_index:
                random.shuffle(self.nodes)
                self.curr_index = 0
                raise StopIteration
            curr_node = int(self.nodes[self.curr_index])
            nodes = [curr_node]
            for j in range(self.walk_length):
                if len(self.graph.neighbors[curr_node]) > 0:
                    next_node = int(random.choice(list(self.graph.neighbors[curr_node])))
                    if next_node not in nodes:
                        nodes.append(next_node)
                        curr_node = next_node
                else:
                    break
            self.curr_index += 1
            nodes = [str(k) for k in nodes]
            return nodes

            
        

default_logger = print


# node的inedx从1开始
def file_to_graph(
                node_path, 
                edge_path,
                label_path="",
                is_weighted=False, 
                is_directed=False):
    with open(node_path,'r') as F:
        L = F.read().split()
        L = [i.strip() for i in L if i.strip()!=""]
        node_number = max([int(i) for i in L])
    adjacent_matrix = dok_matrix((node_number, node_number), dtype=int)
    for i in range(node_number):
        adjacent_matrix[i,i] = 1
    with open(edge_path, 'r') as F:
        edge_number = 0
        L = F.read().split("\n")
        L = [i.strip().split(",") for i in L if i.strip()!=""]
        if not is_directed:
            if not is_weighted:
                for line in L:
                    x = int(line[0]) - 1
                    y = int(line[1]) - 1
                    adjacent_matrix[x,y] = 1
                    adjacent_matrix[y,x] = 1
            else:
                for line in L:
                    x = int(line[0]) - 1
                    y = int(line[1]) - 1
                    z = int(line[2])
                    adjacent_matrix[x,y] = z
                    adjacent_matrix[y,x] = z
        else:
            if not is_weighted:
                k = 0
                for line in L:
                    x = int(line[0]) - 1
                    y = int(line[1]) - 1
                    adjacent_matrix[x,y] = 1
            else:
                for line in L:
                    x = int(line[0]) - 1
                    y = int(line[1]) - 1
                    z = int(line[2])
                    adjacent_matrix[x,y] = z
    adjacent_matrix = adjacent_matrix.tocsr()
    edge_number = adjacent_matrix.count_nonzero()

    label = [set() for i in range(node_number)]
    label_dict = []
    if label_path != "" and label_path != None:
        with open(label_path,'r') as F:
            S = F.read().split("\n")
            L = [i.strip() for i in S if i!=""]
            L = [i.split(",") for i in L]
            for x,y in L:
                x = int(x) - 1
                y = y.strip()
                if y in label_dict:
                    label[x].add(label_dict.index(y))
                else:
                    label_dict.append(y)
                    label[x].add(label_dict.index(y))
    return adjacent_matrix,node_number,edge_number,label,label_dict
        



class Graph:
    def __init__(self, 
                data                = [],
                label               = [],
                label_dict          = [],
                adjacement_matrix   = [],
                other_matrix        = {},
                node_number         = 0,
                edge_number         = 0,
                is_weighted         = False,
                is_directed         = False,
                logger              = default_logger):
        self.data                   = data
        self.label                  = label
        self.label_dict             = label_dict
        self.adjacent_matrix        = adjacement_matrix
        self.other_matrix           = other_matrix
        self.node_number            = node_number
        self.edge_number            = edge_number
        self.is_weighted            = is_weighted
        self.is_directed            = is_directed
        self.edges                  = []
        self.weights                = []
        self.neighbors              = []
        self.logger                 = logger
        self.order                  = numpy.arange(node_number)
        self.epoch_end              = False
        self.start                  = 0
        if self.logger == None:
            self.logger = default_logger

        self.logger("Graph First Init %d Nodes In This Graph"%self.node_number)
        self.logger("Graph First Init %d Edges In This Graph"%self.edge_number)
        self.logger("Graph Init Done")
        self.__set_graph()


    def __set_graph(self):
        self.__set_path()
        self.__set_neighbors()
    def __set_path(self):
        try:
            x = self.adjacent_matrix.nonzero()
            self.edges = [(i,j) for i,j in zip(x[0],x[1])]
            self.weights = [i for i in self.adjacent_matrix.data]
            if self.is_directed:
                index = [i!=j for i,j in self.edges]
            else:
                index = [i>j for i,j in self.edges] 
            self.edges = [i for i,j in zip(self.edges, index) if j]
            self.weights = [i for i,j in zip(self.weights, index) if j]

            # sample_rate = 0.002
            # t0 = time.time()
            # edges = [(i,j) for i in range(self.node_number) for j in range(self.node_number) if i > j]
            # data = [
            #             ((i,j),self.adjacent_matrix[i,j]) 
            #             for i,j in edges 
            #             if (
            #                 self.adjacent_matrix[i,j] > 0 or 
            #                 (self.adjacent_matrix[i,j] == 0 and random.random()< sample_rate)
            #             )
            #         ]
            # self.edges = [i for i,j in data]
            # self.weights = [j for i,j in data]
            # print("Graph Edges %d; Weights %d; Cost %d"%(len(self.edges), len(self.weights), int(time.time()-t0)))

            self.logger("Graph Set Path Done")
        except Exception as e:
            self.logger("Graph Set Path Failed: "+str(e))
    def __set_neighbors(self):
        self.neighbors = [set() for i in range(self.node_number)]
        if not self.is_directed:
            for i,j in self.edges:
                self.neighbors[i].add(j)
                self.neighbors[j].add(i)
        else:
            for i,j in self.edges:
                self.neighbors[i].add(j)

    def init_from_file(self,
                node_path, 
                edge_path, 
                label_path  = "",
                is_weighted = False, 
                is_directed = False,
                data        = []):
        self.logger("Graph Init From CSV File Begin")
        adjacent_matrix,node_number,edge_number,label,label_dict = \
        file_to_graph(node_path, edge_path, label_path, is_weighted, is_directed)
        self.adjacent_matrix    = adjacent_matrix
        self.node_number        = node_number
        self.edge_number        = edge_number
        self.label              = label
        self.label_dict         = label_dict
        self.is_weighted        = is_weighted
        self.is_directed        = is_directed
        self.order              = numpy.arange(node_number)
        if data != [] and self.data == []:
            self.data = data
        else:
            self.data = self.adjacent_matrix
        self.logger("Graph Init From CSV %d Nodes In This Graph"%self.node_number)
        self.logger("Graph Init From CSV %d Edges In This Graph"%self.edge_number)
        self.logger("Graph Init From CSV File Succ")
        self.__set_graph()


        
    def init_from_mat(self, graph_path):
        self.logger("Graph Init From MAT File Begin")
        with shelve.open(graph_path) as data:
            self.edge_number        = data["edge_number"]
            self.node_number        = data["node_number"]
            self.label              = data["label"]
            self.label_dict         = data["label_dict"]
            self.is_weighted        = bool(data["is_weighted"])
            self.is_directed        = bool(data["is_directed"])
            self.edges              = data["edges"]
            self.weights            = data["weights"]
            self.neighbors          = data["neighbors"]
        data = io.loadmat(self.get_mat_path(graph_path))
        self.adjacent_matrix    = data["adjacent_matrix"]
        self.data               = data["data"]
        for key in data.keys():
            if key == "adjacent_matrix":
                self.adjacent_matrix    = data[key]
            elif key == "data":
                self.data               = data[key]
            else:
                self.other_matrix[key]  = data[key]
        self.order              = numpy.arange(self.node_number)
        self.logger("Graph Init From MAT %d Nodes In This Graph"%self.node_number)
        self.logger("Graph Init From MAT %d Edges In This Graph"%self.edge_number)
        self.logger("Graph Init From MAT File Succ")
        self.__set_graph()

    

    def save_to_mat(self, graph_path):
        mat_path = self.get_mat_path(graph_path)
        data = {
            "edge_number"       :   self.edge_number,
            "node_number"       :   self.node_number,
            "label"             :   self.label,
            "label_dict"        :   self.label_dict,
            "is_weighted"       :   self.is_weighted,
            "is_directed"       :   self.is_directed,
            "edges"             :   self.edges,
            "weights"           :   self.weights,
            "neighbors"         :   self.neighbors
        }
        with shelve.open(graph_path) as file:
            for key in data.keys():
                file[key] = data[key]
        d = {}
        d["adjacent_matrix"] = self.adjacent_matrix
        d["data"]            = self.data
        for key in self.other_matrix:
            d["key"] = self.other_matrix[key]
        io.savemat(mat_path, d)
        self.logger("Graph Save To %s Succ"%graph_path)
        
    def get_mat_path(self, graph_path):
        return graph_path + ".mat"
        



    # 一种是边采样，一种是点采样
    # 边采样适用于图计算
    # 点采样适用于一般向量降维

    def sample_with_subgraph(self, batch_size):
        self.order = numpy.arange(self.node_number)
        numpy.random.shuffle(self.order)
        self.start      = 0
        self.epoch_end  = False
        while not self.epoch_end:
            index            = self.__subgraph(index = self.order[self.start], size = batch_size)
            if (self.start >= self.node_number-1):
                self.epoch_end = True
                self.start = 0
            else:
                self.start += 1
            if len(index) > 0:
                yield index

    def sample_without_subgraph(self, batch_size):
        self.order = numpy.arange(self.node_number)
        numpy.random.shuffle(self.order)
        self.start      = 0
        self.epoch_end  = False
        while not self.epoch_end:
            end              = min(self.start + batch_size,self.node_number)
            index            = self.order[self.start:end]
            if (self.start >= self.node_number-1):
                self.epoch_end = True
                self.start = 0
            else:
                self.start = end
            if len(index) > 0:
                yield index

    def __subgraph(self, index, size):
        t = [index]
        while True:
            t = self.adjacent_matrix[t,:].nonzero()[1]
            if t.shape[0] >= size:
                numpy.random.shuffle(t)
                t = t[0:size]
                break
        return t

    def sample_pairwise(self, batch_size):
        edge_order = numpy.arange(len(self.edges))
        numpy.random.shuffle(edge_order)
        self.start      = 0
        self.epoch_end  = False
        ids1 = []
        ids2 = []
        N = len(self.edges)
        while True:
            end = min(N, self.start+batch_size)
            if self.start >= N - 1:
                self.start = 0
                self.epoch_end = True
                break
            X = [self.edges[i] for i in range(self.start,end)]
            weights = [self.weights[i] for i in range(self.start,end)]
            ids1 = [i[0] for i in X]
            ids2 = [i[1] for i in X]
            self.start = end
            yield ids1,ids2,weights
  


    def cal_jaccard_matrix(self):
        jaccard_matrix = kneighbors_graph(
                            self.adjacent_matrix, 
                            self.node_number, 
                            metric='cosine',
                            mode='distance', 
                            include_self=True)
        jaccard_matrix =  csr_matrix(numpy.ones(jaccard_matrix.shape)) - jaccard_matrix
        # for i in range(self.node_number):
        #     for j in range(i, self.node_number):
        #         if i == j:
        #             jaccard_matrix[i,j],jaccard_matrix[j,i] = 1,1
        #         elif self.adjacent_matrix[i,j] == 1:
        #             jaccard_matrix[i,j],jaccard_matrix[j,i] = 1,1
        #         else:
        #             listi = self.adjacent_matrix[i,:].nonzero()[1]
        #             listj = self.adjacent_matrix[j,:].nonzero()[1]
        #             x = cal_jaccard(i,j, listi,listj)
        #             print(i,j,x)
        #             jaccard_matrix[i,j],jaccard_matrix[j,i] = x,x
        self.other_matrix["jaccard_matrix"] = jaccard_matrix
        jaccard_matrix.tocsr()
        self.other_matrix["jaccard_matrix"] = jaccard_matrix


def cal_jaccard(i,j, listi,listj):
    ai,aj = set(listi),set(listj)
    aij_union = ai.union(aj)
    if len(aij_union) == 0:
        return 0
    ai.remove(i)
    aj.remove(j)
    aij_inter = ai.intersection(aj)
    return float(len(aij_inter))/len(aij_union)




if __name__ == "__main__":
    node_path   = "D:\\Code\\Python3\\network_embedding\\data\\Citeseer\\nodes.csv"
    edge_path   = "D:\\Code\\Python3\\network_embedding\\data\\Citeseer\\edges.csv"
    label_path  = "D:\\Code\\Python3\\network_embedding\\data\\Citeseer\\group-edges.csv"
    is_weighted = False
    is_directed = False
    graph_path  = "graph"
    data        = []
    graph = Graph()
    try:
        graph.init_from_mat(graph_path)
    except:
        graph.init_from_file(node_path,edge_path,label_path,is_weighted,is_directed)
        graph.save_to_mat(graph_path)
    
    # for index in graph.sample_without_subgraph(1000):
        # print(index.shape,numpy.sum(index))
    