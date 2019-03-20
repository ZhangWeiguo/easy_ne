import math
import numpy
from numpy import linalg as la
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from scipy import io

def reduce_dim(d):
    tsne = TSNE(n_iter=500)
    embedding = tsne.fit_transform(d)
    return embedding
class GrapRep:
    def __init__(self, step, embedding_size = 64, logger = print):
        self.step = step
        self.embedding_size = embedding_size
        self.node_size = -1
        self.logger = logger

    def train(self, adj):
        self.node_size = adj.shape[0]
        a = numpy.matrix(numpy.identity(self.node_size))
        embedding = numpy.zeros((self.node_size, int(self.embedding_size*self.step)))
        for i in range(self.step):
            self.logger("Train GrapReq Step %d: Begin"%i)
            a = numpy.dot(a, adj)
            self.logger("Train GrapReq Step %d: Cal Prob"%i)
            p = self.__get_probmat(a)
            self.logger("Train GrapReq Step %d: Cal SVD"%i)
            r = self.__get_svd(p)
            self.logger("Train GrapReq Step %d: Normalize"%i)
            r = normalize(r, axis=1, norm='l2')
            self.logger("Train GrapReq Step %d: Cal Embedding"%i)
            embedding[:, self.embedding_size*i:self.embedding_size*(i+1)] = r[:, :]
            self.logger("Train GrapReq Step %d: Finish"%i)
        return embedding

    def __get_probmat(self, a):
        p = numpy.log(a / numpy.tile(
            numpy.sum(a, axis = 0), (self.node_size, 1)
        )) - numpy.log(1.0 / self.node_size)
        p[p < 0 ] = 0
        p[p == numpy.nan] = 0
        return p

    def __get_svd(self, a):
        model = TruncatedSVD(self.embedding_size)
        return  model.fit_transform(a)


