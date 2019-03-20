import sys
sys.path.append("..")
from basic.logger import Logger
from basic.config_parse import IniConfiger
from model.deepwalk import DeepWalk
from graph.graph import Graph,deepwalk_sample
import numpy

data_name = sys.argv[1]
logger = Logger("DeepWalk","DEEPWALK_%s.log"%data_name,"None")
logger.info("Config Begin")
config = IniConfiger("../config/%s.ini"%data_name)
is_weighted         = bool(config.get("graph","is_weighted","int"))
is_directed         = bool(config.get("graph","is_directed","int"))
node_path           = config.get("graph","node_path")
edge_path           = config.get("graph","edge_path")
label_path          = config.get("graph","label_path")
graph_path          = config.get("graph","graph_path")

walk_length          = config.get("deepwalk","walk_length","int")
embedding_size       = config.get("deepwalk","embedding_size","int")
window               = config.get("deepwalk","window","int")
workers              = config.get("deepwalk","workers","int")
epochs               = config.get("deepwalk","epochs","int")
model_path           = config.get("deepwalk","model_path")
embedding_path       = config.get("deepwalk","embedding_path")
logger.info("Config Done")


logger.info("Graph Construct Begin")
graph = Graph(logger=logger.info)
try:
    graph.init_from_mat(graph_path)
except:
    graph.init_from_file(node_path,edge_path,label_path,is_weighted,is_directed)
    graph.save_to_mat(graph_path)
logger.info("Graph Construct Done")


model = DeepWalk(
    walk_length,
    embedding_size,
    window,
    workers,
    epochs = epochs,
    logger = logger.info, 
    model_path = model_path, 
    embedding_path = embedding_path)

try:
    model.restore_model(model_path)
    logger.info("Model Init From Model File Succ")
except:
    logger.info("Model Init From Model File Fail")
logger.info("Model Training Begin")

model.train(deepwalk_sample, graph)

logger.info("Model Training Done")    
model.save_model(model_path)
logger.info("Model Saved To %s"%model_path)