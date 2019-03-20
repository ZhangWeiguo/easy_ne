import sys
from scipy import io
from scipy.sparse import csr_matrix
from metric.metric_net_restruct import metric_map,metric_precision_ks
from sklearn.neighbors import kneighbors_graph
import numpy
import json

data_name = sys.argv[1]
model_name = sys.argv[2]
graph_path      = "data/%s/graph.mat"%data_name
embedding_path  = "data/%s/embedding_%s.mat"%(data_name, model_name)

data            = io.loadmat(graph_path)
adj_matrix      = data["adjacent_matrix"]
embedding       = io.loadmat(embedding_path)["embedding"]


adj_matrix_e    = kneighbors_graph(embedding, embedding.shape[0], 
                            metric='cosine',
                            mode='distance', 
                            include_self=True)

m               = metric_map(adj_matrix_e, adj_matrix)
print(m)
