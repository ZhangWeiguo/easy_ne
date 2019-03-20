# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy,time
from scipy import io


class LINE(object):
    def __init__(self,
                embedding_size       = 100,
                node_number          = 1000,
                batch_size           = 200, 
                epochs               = 100,
                min_re_err           = 0.0001,
                learning_rate        = 0.06,
                learning_decay       = False,
                learning_decay_rate  = 0.99,
                learning_decay_steps = 50,
                logger               = None,
                model_path           = "model",
                embedding_path       = "embedding"):

        self.node_number        = node_number
        self.embedding_size     = embedding_size
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.min_re_err         = min_re_err
        self.learning_rate      = learning_rate
        self.learning_decay     = learning_decay
        self.init               = False

        # decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
        if self.learning_decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step = self.global_step, 
                decay_steps = learning_decay_steps, 
                decay_rate  = learning_decay_rate)

        self.input_ids1    = tf.placeholder(tf.int32, [None])
        self.input_ids2    = tf.placeholder(tf.int32, [None])
        self.distance      = tf.placeholder(tf.float32, [None])


        def default_logger(s):
            print(s)
 
        self.__cal_loss()

        self.session = tf.Session()

        self.model_path         = model_path
        self.embedding_path     = embedding_path
        if logger == None:
            self.logger = default_logger
        else:
            self.logger = logger


    def __cal_loss(self):
        self.embedding  = tf.get_variable(
                    name="embedding",
                    shape=[self.node_number,self.embedding_size],
                    initializer=tf.contrib.layers.xavier_initializer())
        self.embedding_context = tf.get_variable(
                    name="embedding_context",
                    shape=[self.node_number,self.embedding_size],
                    initializer=tf.contrib.layers.xavier_initializer())
        
        self.input_1            = tf.nn.embedding_lookup(self.embedding, self.input_ids1)
        self.input_2            = tf.nn.embedding_lookup(self.embedding, self.input_ids2)
        self.input_2_context    = tf.nn.embedding_lookup(self.embedding_context, self.input_ids2)
        self.loss_2nd = -tf.reduce_mean(self.distance * tf.log_sigmoid(
            tf.reduce_sum(tf.multiply(self.input_1, self.input_2_context), axis=1)))
        self.loss_1st = -tf.reduce_mean(self.distance * tf.log_sigmoid(
            tf.reduce_sum(tf.multiply(self.input_1, self.input_2), axis=1)))
        self.loss = self.loss_1st
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
           self.loss,global_step=self.global_step)


    # train all data by batch
    def train(self, sample_fun, graph, batch_size):
        if not self.init:
            self.session.run(tf.global_variables_initializer())
        self.tensor_writer = tf.summary.FileWriter("TensorGraph.log", tf.get_default_graph())
        batch_size       = self.batch_size
        model_path       = self.model_path
        embedding_path   = self.embedding_path
        current_epoch    = 0
        last_loss        = 0
        loss             = 0
        for k,(ids1,ids2,distance) in enumerate(sample_fun(graph,batch_size)):
            if k >= 100:
                break
            t_loss = self.get_loss(ids1,ids2,distance)
            loss += t_loss
        self.logger("Epoch %3d Loss(All=%5.1f)"%(current_epoch, loss))
        while True:
            if current_epoch < self.epochs:
                current_epoch += 1
                embedding = None
                node_number = self.node_number
                start = 0
                for ids1,ids2,distance in sample_fun(graph,batch_size):
                    t0 = time.time()
                    loss = self.__fit(ids1,ids2,distance)
                    train_time = time.time() - t0
                    learning_rate = self.__get_learning_rate()
                    self.logger("Epoch(%3d) LR(%1.5f) Time(Train=%1.5f) Rate(%5d/%5d) Loss(All=%5.1f)"%(
                        current_epoch, learning_rate, train_time,start, node_number, loss))
                    start += batch_size
                model_path_epoch = model_path + ".%s"%str(current_epoch)
                embedding_path_epoch = embedding_path + ".%s"%str(current_epoch)
                # self.save_model(model_path_epoch)
                self.save_model(model_path)

                loss = 0
                for k,(ids1,ids2,distance) in enumerate(sample_fun(graph,batch_size)):
                    if k >= 100:
                        break
                    t_loss = self.get_loss(ids1,ids2,distance)
                    loss += t_loss
                embedding = self.get_embedding()
                # io.savemat(embedding_path_epoch, {"embedding":embedding})
                io.savemat(embedding_path,{"embedding":embedding})
                self.logger("Epoch %3d Loss(All=%5.1f)"%(current_epoch, loss))
                if last_loss > 0 and (last_loss - loss)/last_loss <= self.min_re_err and loss < last_loss:
                    self.logger("Stop Iter As Loss Reduce Smaller Than MinReError")
                    return
            else:
                break
        self.tensor_writer.close()

    def __get_feed_dict(self, input_ids1, input_ids2, distance):
        return {self.input_ids1     : input_ids1, 
                self.input_ids2     : input_ids2,
                self.distance       : distance}

    # train one batch
    def __fit(self, input_ids1, input_ids2, distance):
        feed_dict = self.__get_feed_dict(input_ids1, input_ids2, distance)
        loss, _ = self.session.run((self.loss,self.optimizer), feed_dict = feed_dict)
        return loss

    def __get_learning_rate(self):
        if self.learning_decay:
            return self.session.run(self.learning_rate)
        else:
            return self.learning_rate

    def get_loss(self, input_ids1, input_ids2, distance):
        feed_dict = self.__get_feed_dict(input_ids1, input_ids2, distance)
        return self.session.run(self.loss, feed_dict = feed_dict)

    def get_embedding(self):
        return self.session.run(
             tf.nn.embedding_lookup(self.embedding, [i for i in range(self.node_number)])
        )


    def close(self):
        self.session.close()

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)
        self.init = True
    

        

