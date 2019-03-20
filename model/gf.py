# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy,time
from scipy import io

# Graph Factorization 

class GF:
    def __init__(self, node_number, embedding_size, epochs = 30, batch_size=100, learing_rate=0.051, logger=None):
        self.node_number        = node_number
        self.embedding_size     = embedding_size
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.learning_rate      = learing_rate
        def default_logger(s):
            print(s)
        if logger == None:
            self.logger = default_logger
        else:
            self.logger = logger

    def train(self,sample_fun, graph, init_embedding=None):
        try:
            self.embedding = tf.Variable(init_embedding, name="embedding", dtype=tf.float32)
            self.logger("Init Embedding From Init Succ")
        except:
            self.logger("Init Embedding From Init Fail")
            self.embedding = tf.get_variable(
                name="embedding",
                shape=[self.node_number,self.embedding_size],dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        self.input_ids1    = tf.placeholder(tf.int32, [None])
        self.input_ids2    = tf.placeholder(tf.int32, [None])
        self.distance      = tf.placeholder(tf.float32, [None])
        self.input_1       = tf.nn.embedding_lookup(self.embedding, self.input_ids1)
        self.input_2       = tf.nn.embedding_lookup(self.embedding, self.input_ids2)
        self.loss          = -tf.reduce_sum(tf.sigmoid(tf.reduce_sum(tf.multiply(self.input_1, self.input_2), axis=1)))
        # self.loss          = tf.reduce_sum(tf.square(self.distance - tf.reduce_sum(tf.square((self.input_1-self.input_2)),keep_dims=True)))
        # self.loss          = tf.reduce_sum(self.distance * tf.reduce_sum(tf.square((self.input_1-self.input_2)),keep_dims=True))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        for current_epoch in range(self.epochs):
            loss = 0
            for ids1,ids2,distance in sample_fun(graph, self.batch_size):
                losst = self.__fit(ids1,ids2,distance)
                loss += losst
            self.logger("Epoch(%3d) Loss(All=%5.1f)"%(current_epoch, loss))
        return self.get_embedding()
            

    def __get_feed_dict(self, input_ids1, input_ids2, distance):
        return {self.input_ids1     : input_ids1, 
                self.input_ids2     : input_ids2,
                self.distance       : distance}

    # train one batch
    def __fit(self, input_ids1, input_ids2, distance):
        feed_dict = self.__get_feed_dict(input_ids1, input_ids2, distance)
        loss, _ = self.session.run((self.loss,self.optimizer), feed_dict = feed_dict)
        return loss

    def get_embedding(self):
        return self.session.run(self.embedding)


