import sys
sys.path.append("..")
from basic.logger import Logger
from basic.config_parse import IniConfiger
from model.sdne import SDNE
from graph.graph import Graph,sdne_sample
import numpy

data_name = sys.argv[1]

logger = Logger("SDNE","SDNE_%s.log"%data_name,"None")
logger.info("Config Begin")

config = IniConfiger("../config/%s.ini"%data_name)
is_weighted         = bool(config.get("graph","is_weighted","int"))
is_directed         = bool(config.get("graph","is_directed","int"))
node_path           = config.get("graph","node_path")
edge_path           = config.get("graph","edge_path")
label_path          = config.get("graph","label_path")
graph_path          = config.get("graph","graph_path")

struct               = list(map(int,config.get("sdne","struct").split(",")))
alpha1               = config.get("sdne","alpha1","float")
alpha2               = config.get("sdne","alpha2","float")
beta                 = config.get("sdne","beta","float")
nu1                  = config.get("sdne","nu1","float")
nu2                  = config.get("sdne","nu2","float")
batch_size           = config.get("sdne","batch_size","int")
epochs               = config.get("sdne","epochs","int")
min_re_err           = config.get("sdne","min_re_err","float")
learning_rate        = config.get("sdne","learning_rate","float")
learning_decay       = bool(config.get("sdne","learning_decay","int"))
learning_decay_rate  = config.get("sdne","learning_decay_rate","float")
learning_decay_steps = config.get("sdne","learning_decay_steps","float")
model_path           = config.get("sdne","model_path")
embedding_path       = config.get("sdne","embedding_path")
logger.info("Config Done")


logger.info("Graph Construct Begin")
graph = Graph(logger=logger.info)
try:
    graph.init_from_mat(graph_path)
except:
    graph.init_from_file(node_path,edge_path,label_path,is_weighted,is_directed)
    graph.save_to_mat(graph_path)
logger.info("Graph Construct Done")


struct_t = [graph.node_number]
struct_t.extend(struct)
struct = struct_t

model = SDNE(
    struct,alpha1,alpha2,
    beta,nu1,nu2,batch_size,
    epochs,min_re_err,learning_rate,
    learning_decay,learning_decay_rate,
    learning_decay_steps,logger.info,
    model_path,embedding_path)
try:
    model.restore_model(model_path)
    logger.info("Model Init From Model File Succ")
except:
    logger.info("Model Init From Model File Fail")
logger.info("Model Training Begin")
model.train(sdne_sample,graph, batch_size)

w = model.get_w()
b = model.get_b()
for i in w:
    print(i, w[i].shape, w[i])
for i in b:
    print(i, b[i].shape ,b[i])
for j in range(5):
    i = numpy.random.randint(0, 100)
    x = graph.adjacent_matrix[i:i+1,:].todense()
    index = numpy.arange(5)
    h = model.get_embedding(x)
    print("Before Embedding:", x[0,index])
    print("After Embedding:", h[0,index])
    x_r = model.get_reconstruct(graph.adjacent_matrix[i:i+1,:].todense())
    print("After Restruct:", x_r[0,index])
    print("Error:", numpy.sum(x),numpy.sum(x_r))

logger.info("Model Training Done")    
model.save_model(model_path)
logger.info("Model Saved To %s"%model_path)