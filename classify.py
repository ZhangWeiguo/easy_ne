from metric.metric_classifier import MultiLabelClassifier as Classifier
from sklearn.model_selection import train_test_split
from graph.graph import Graph
from scipy import io
import numpy
import json
import sys
from sklearn.neighbors import kneighbors_graph

data_name = sys.argv[1]
model_name = sys.argv[2]
train_size = float(sys.argv[3])
graph_path      = "data/%s/graph"%data_name
embedding_path  = "data/%s/embedding_%s.mat"%(data_name, model_name)


graph           = Graph()
graph.init_from_mat(graph_path)
data            = io.loadmat(graph_path)
adj_matrix      = graph.adjacent_matrix
label           = graph.label
if model_name == 'subsdne':
    embedding       = adj_matrix
    embedding       = kneighbors_graph(embedding, embedding.shape[0], metric='cosine', mode='distance', include_self=True)
elif model_name == "sdne":
    embedding       = adj_matrix
else:
    embedding       = io.loadmat(embedding_path)["embedding"]
index           = [i for i in range(graph.node_number) if len(label[i])>0]
embedding       = embedding[index,:]
label           = [label[i] for i in index]
print(embedding.shape,len(label))


train_feature,test_feature,train_label,test_label = train_test_split(embedding, label, train_size=train_size, random_state=0)

model           = Classifier()
model.train(train_feature, train_label)
print("Train %5d  Test %5d"%(len(train_label),len(test_label)))
print("Train Macro F1 Score: ", model.macro_f1(train_feature, train_label))
print("Train Micro F1 Score: ", model.micro_f1(train_feature, train_label))
print("Test  Macro F1 Score: ", model.macro_f1(test_feature, test_label))
print("Test  Micro F1 Score: ", model.micro_f1(test_feature, test_label))