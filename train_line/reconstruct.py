import sys
sys.path.append("..")
from basic.logger import Logger
from basic.config_parse import IniConfiger
from model.line import LINE
from graph.graph import Graph,line_sample
import numpy

data_name = sys.argv[1]

logger = Logger("LINE","LINE_%s.log"%data_name,"None")
logger.info("Config Begin")

config = IniConfiger("../config/%s.ini"%data_name)
is_weighted         = bool(config.get("graph","is_weighted","int"))
is_directed         = bool(config.get("graph","is_directed","int"))
node_path           = config.get("graph","node_path")
edge_path           = config.get("graph","edge_path")
label_path          = config.get("graph","label_path")
graph_path          = config.get("graph","graph_path")


embedding_size       = config.get("line","embedding_size","int")
batch_size           = config.get("line","batch_size","int")
epochs               = config.get("line","epochs","int")
min_re_err           = config.get("line","min_re_err","float")
learning_rate        = config.get("line","learning_rate","float")
learning_decay       = bool(config.get("line","learning_decay","int"))
learning_decay_rate  = config.get("line","learning_decay_rate","float")
learning_decay_steps = config.get("line","learning_decay_steps","float")
model_path           = config.get("line","model_path")
embedding_path       = config.get("line","embedding_path")
logger.info("Config Done")


logger.info("Graph Construct Begin")
graph = Graph(logger=logger.info)
try:
    graph.init_from_mat(graph_path)
except:
    graph.init_from_file(node_path,edge_path,label_path,is_weighted,is_directed)
    graph.save_to_mat(graph_path)
logger.info("Graph Construct Done")


node_number = graph.node_number

model = LINE(
    embedding_size,node_number,batch_size,
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
model.train(line_sample,graph, batch_size)


logger.info("Model Training Done")    
model.save_model(model_path)
logger.info("Model Saved To %s"%model_path)