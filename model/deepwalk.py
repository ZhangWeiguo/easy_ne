# -*- encoding:utf-8 -*-
from gensim.models import Word2Vec
import random
import numpy
from scipy import io,sparse
class DeepWalk:
    def __init__(self, 
                walk_length         = 20,
                embedding_size      = 128,
                window              = 5,
                workers             = 8,
                epochs              = 20,
                logger              = print,
                model_path          = "model",
                embedding_path      = "embedding"):
        self.walk_length            = walk_length
        self.embedding_size         = embedding_size
        self.window                 = window
        self.workers                = workers
        self.epochs                 = epochs
        self.word2vec               = None
        self.embedding              = None
        self.model_path             = model_path
        self.embedding_path         = embedding_path
        self.logger                 = logger

    def train(self, sample_fun, graph):
        conf                = {}
        sentences           = sample_fun(graph, self.walk_length)
        conf["sentences"]   = sentences
        conf["min_count"]   = 1
        conf["size"]        = self.embedding_size
        conf["workers"]     = self.workers
        conf["window"]      = self.window
        conf["sg"]          = 1
        conf["hs"]          = 1
        conf["negative"]    = 0
        conf["iter"]        = self.epochs
        self.word2vec       = Word2Vec(**conf)
        self.embedding      = sparse.dok_matrix((graph.node_number, self.embedding_size))
        for i in range(graph.node_number):
            self.embedding[i] = self.word2vec.wv[str(i)]
        io.savemat(self.embedding_path, {"embedding":self.embedding})
        self.save_model(self.model_path)

    def save_model(self, model_path):
        self.word2vec.save(model_path)
    
    def restore_model(self, model_path):
        self.word2vec = Word2Vec.load(model_path)

    def get_embedding(self):
        return self.embedding



