# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy,time
from scipy import io
from sklearn.metrics import pairwise_distances
from .rbm import RBM


class AENE(object):
    def __init__(self,
                struct1              = [100, 10],
                struct2              = [100, 10],
                alpha                = 1e-1,
                alpha1               = 1,
                alpha2               = 1,
                beta1                = 5.0,
                beta2                = 5.0,
                nu1                  = 1e-5,
                nu2                  = 1e-4,
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
        # 保持两个结构长度一致
        self.struct1            = struct1
        self.struct2            = struct2
        if len(self.struct1) != len(self.struct2):
            raise Exception("Struct Length Must Be Same")
        self.layers             = len(self.struct1)
        self.alpha1             = alpha1
        self.alpha2             = alpha2
        self.alpha              = alpha
        self.beta1              = beta1
        self.beta2              = beta2
        self.nu1                = nu1
        self.nu2                = nu2
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

        # x1为一阶距离 x2为二阶距离 
        self.x1a                = tf.placeholder(tf.float32, [None, self.struct1[0]], name="x1a")
        self.x2a                = tf.placeholder(tf.float32, [None, self.struct2[0]], name="x2a")
        self.x1b                = tf.placeholder(tf.float32, [None, self.struct1[0]], name="x1b")
        self.x2b                = tf.placeholder(tf.float32, [None, self.struct2[0]], name="x2b")
        self.weight             = tf.placeholder(tf.float32, [None], name="weigth")



        def default_logger(s):
            print(s)
 
        self.B1a = tf.add(tf.ones_like(self.x1a),tf.sign(self.x1a)*(self.beta1-1), name="B1a")
        self.B2a = tf.add(tf.ones_like(self.x2a),tf.sign(self.x2a)*(self.beta2-1), name="B2a")

        self.B1b = tf.add(tf.ones_like(self.x1b),tf.sign(self.x1b)*(self.beta1-1), name="B1b")
        self.B2b = tf.add(tf.ones_like(self.x2b),tf.sign(self.x2b)*(self.beta2-1), name="B2b")

        self.ha                 = None
        self.hb                 = None
        self.x1a_reconstruct    = None
        self.x2a_reconstruct    = None
        self.x1b_reconstruct    = None
        self.x2b_reconstruct    = None
        self.w1                 = {}
        self.b1                 = {}
        self.w2                 = {}
        self.b2                 = {}
        self.__make_compute_graph()
        self.__cal_loss()

        self.session = tf.Session()

        self.model_path         = model_path
        self.embedding_path     = embedding_path
        if logger == None:
            self.logger = default_logger
        else:
            self.logger = logger


        

    def __make_compute_graph(self, active_func = tf.nn.sigmoid):
        def encoder(x1, x2):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                x1 = active_func(tf.matmul(x1, self.w1[name]) + self.b1[name])
                x2 = active_func(tf.matmul(x2, self.w2[name]) + self.b2[name])
            return x1,x2
        def decoder(x):
            x1 = x
            x2 = x
            for i in range(self.layers - 1):
                name = "decoder" + str(i)
                x1 = active_func(tf.matmul(x1, self.w1[name]) + self.b1[name])
                x2 = active_func(tf.matmul(x2, self.w2[name]) + self.b2[name])
            return x1,x2
        struct = self.struct1
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.w1[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = "w1"+name)
            self.b1[name] = tf.Variable(tf.random_normal([struct[i+1]]), name = "b1"+name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.w1[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = "w1"+name)
            self.b1[name] = tf.Variable(tf.random_normal([struct[i+1]]), name = "b1"+name)
        struct.reverse()


        struct = self.struct2
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.w2[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = "w2"+name)
            self.b2[name] = tf.Variable(tf.random_normal([struct[i+1]]), name = "b2"+name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.w2[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = "w2"+name)
            self.b2[name] = tf.Variable(tf.random_normal([struct[i+1]]), name = "b2"+name)
        struct.reverse()
    
        x1a,x2a = encoder(self.x1a, self.x2a)
        x1b,x2b = encoder(self.x1b, self.x2b)
        self.ha = x1a + x2a
        self.hb = x1b + x2b
        self.x1a_reconstruct,self.x2a_reconstruct = decoder(self.ha)
        self.x1b_reconstruct,self.x2b_reconstruct = decoder(self.hb)

    
        self.rbms1 = []
        self.rbms2 = []
        for i in range(self.layers - 1):
            rbm_unit1 = RBM(
                shape           = [self.struct1[i], self.struct1[i+1]], 
                batch_size      = self.batch_size, 
                learning_rate   = self.learning_rate)
            rbm_unit2 = RBM(
                shape           = [self.struct2[i], self.struct2[i+1]], 
                batch_size      = self.batch_size, 
                learning_rate   = self.learning_rate)
            self.rbms1.append(rbm_unit1)
            self.rbms2.append(rbm_unit2)

    
    def __cal_loss(self):

        self.loss_1st = tf.reduce_sum(
                self.weight * tf.reduce_sum(tf.square((self.ha-self.hb)),keep_dims=True))
        
        self.loss_2nd1 = \
                tf.reduce_sum(tf.square((self.x1a_reconstruct-self.x1a)*self.B1a)) + \
                tf.reduce_sum(tf.square((self.x1b_reconstruct-self.x1b)*self.B1b))

        self.loss_2nd2 = \
                tf.reduce_sum(tf.square((self.x2a_reconstruct-self.x2a)*self.B2a)) + \
                tf.reduce_sum(tf.square((self.x2b_reconstruct-self.x2b)*self.B2b))

        self.loss_reg = None
        for w1i,w2i in zip(self.w1,self.w2):
            loss1_tmp1 = tf.contrib.layers.l1_regularizer(self.nu1)(self.w1[w1i]) + tf.contrib.layers.l1_regularizer(self.nu1)(self.b1[w1i])
            loss1_tmp2 = tf.contrib.layers.l1_regularizer(self.nu1)(self.w2[w2i]) + tf.contrib.layers.l1_regularizer(self.nu1)(self.b2[w2i])

            loss2_tmp1 = self.nu2 * (tf.nn.l2_loss(self.w1[w1i]) + tf.nn.l2_loss(self.b1[w1i]))
            loss2_tmp2 = self.nu2 * (tf.nn.l2_loss(self.w2[w2i]) + tf.nn.l2_loss(self.b2[w2i]))
            if self.loss_reg == None:
                self.loss_reg = loss1_tmp1 + loss1_tmp2 + loss2_tmp1 + loss2_tmp2
            else:
                self.loss_reg += (loss1_tmp1 + loss1_tmp2 + loss2_tmp1 + loss2_tmp2)

        self.loss = self.alpha * self.loss_1st + self.alpha1 * self.loss_2nd1 + self.alpha2 * self.loss_2nd2 + self.loss_reg
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        # self.optimizer_1st = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_1st)
        # self.optimizer_2nd = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_2nd)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
        #    self.loss,global_step=self.global_step)


    # train all data by batch
    def train(self, sample_fun, graph,batch_size):
        if not self.init:
            self.session.run(tf.global_variables_initializer())
            # self.__rbm_init(sample_fun, graph, batch_size)
            self.init = True
        self.tensor_writer = tf.summary.FileWriter("TensorGraph.log", tf.get_default_graph())
        batch_size       = self.batch_size
        model_path       = self.model_path
        embedding_path   = self.embedding_path
        current_epoch    = 0
        last_loss        = 0
        loss,loss_1st,loss_2nd1,loss_2nd2,loss_reg = 0,0,0,0,0
        for x1a, x2a, x1b, x2b, weight,ids1,ids2 in sample_fun(graph,batch_size):
            print(x1a.shape,x2a.shape)
            print(x1b.shape,x2b.shape)
            t_loss,t_loss_1st,t_loss_2nd1,t_loss_2nd2,loss_reg = self.get_loss(x1a, x2a, x1b, x2b, weight)
            loss        += t_loss
            loss_1st    += t_loss_1st
            loss_2nd1   += t_loss_2nd1
            loss_2nd2   += t_loss_2nd2
        self.logger("Epoch %3d Loss(All=%5.1f 1st=%5.1f 2nd1=%5.1f 2nd2=%5.1f reg=%5.1f)"%(
            current_epoch, loss,loss_1st,loss_2nd1,loss_2nd2,loss_reg))
        while True:
            if current_epoch < self.epochs:
                current_epoch += 1
                embedding = None
                node_number = len(graph.edges)
                start = 0
                for x1a, x2a, x1b, x2b, weight,ids1,ids2 in sample_fun(graph,batch_size):
                    t0 = time.time()
                    loss,loss_1st,loss_2nd1,loss_2nd2,loss_reg = self.__fit(x1a, x2a, x1b, x2b, weight)
                    train_time = time.time() - t0
                    learning_rate = self.__get_learning_rate()
                    self.logger("Epoch(%3d) LR(%1.5f) Time(Train=%1.5f) Rate(%5d/%5d) Loss(All=%5.1f 1st=%5.1f 2nd1=%5.1f 2nd2=%5.1f reg=%5.1f)"%(
                        current_epoch, learning_rate, train_time,start, node_number, loss,loss_1st,loss_2nd1, loss_2nd2,loss_reg))
                    start += batch_size
                model_path_epoch = model_path + ".%s"%str(current_epoch)
                embedding_path_epoch = embedding_path + ".%s"%str(current_epoch)
                # self.save_model(model_path_epoch)
                self.save_model(model_path)

                loss,loss_1st,loss_2nd1,loss_2nd2,loss_reg = 0,0,0,0,0
                embedding = self.get_embedding(graph.adjacent_matrix.todense(), graph.data.todense())
                for x1a, x2a, x1b, x2b, weight,ids1,ids2 in sample_fun(graph,batch_size):
                    t_loss,t_loss_1st,t_loss_2nd1,t_loss_2nd2,loss_reg = self.get_loss(x1a, x2a, x1b, x2b, weight)
                    loss        += t_loss
                    loss_1st    += t_loss_1st
                    loss_2nd1   += t_loss_2nd1
                    loss_2nd2   += t_loss_2nd2
                # io.savemat(embedding_path_epoch, {"embedding":embedding})
                io.savemat(embedding_path,{"embedding":embedding})
                self.logger("Epoch %3d Loss(All=%5.1f 1st=%5.1f 2nd1=%5.1f 2nd1=%5.1f reg=%5.1f)"%(
                    current_epoch, loss,loss_1st,loss_2nd1, loss_2nd2,loss_reg))
                if last_loss > 0 and (last_loss - loss)/last_loss <= self.min_re_err and loss < last_loss:
                    self.logger("Stop Iter As Loss Reduce Smaller Than MinReError")
                    return
            else:
                break
        self.tensor_writer.close()

    def __rbm_init(self, sample_fun, graph, batch_size):
        def assign(a, b):
            op = a.assign(b)
            self.session.run(op)
        self.logger("Init Para By RBM Model Begin")
        for i in range(self.layers - 1):
            rbm_unit1 = self.rbms1[i]
            rbm_unit2 = self.rbms2[i]
            for epoch in range(5):
                error = 0
                for x1a, x2a, x1b, x2b, weight,ids1,ids2 in sample_fun(graph,batch_size):
                    for k in range(i):
                        x1a = self.rbms1[k].predict(x1a)
                        x1b = self.rbms1[k].predict(x1b)
                        x2a = self.rbms2[k].predict(x2a)
                        x2b = self.rbms2[k].predict(x2b)
                    error_batch1 = rbm_unit1.fit(x1a)
                    error_batch1 += rbm_unit1.fit(x1b)
                    error_batch2 = rbm_unit2.fit(x2a) 
                    error_batch2 += rbm_unit2.fit(x2b)
                    error += (error_batch1+error_batch2)
                self.logger("%d Layer: Rbm Epochs %3d Error: %5.6f"%(i,epoch,error))

            W1, bv1, bh1 = rbm_unit1.get_para()
            W2, bv2, bh2 = rbm_unit2.get_para()
            name = "encoder" + str(i)
            assign(self.w1[name], W1)
            assign(self.b1[name], bh1)
            assign(self.w2[name], W2)
            assign(self.b2[name], bh2)
            name = "decoder" + str(self.layers-i-2)
            assign(self.w1[name], W1.transpose())
            assign(self.b1[name], bv1)
            assign(self.w2[name], W2.transpose())
            assign(self.b2[name], bv2)
        self.logger("Init Para By RBM Model Done")

    def __get_feed_dict(self, x1a, x2a, x1b, x2b, weight):
        return {self.x1a                :   x1a,
                self.x2a                :   x2a,
                self.x1b                :   x1b,
                self.x2b                :   x2b,
                self.weight             :   weight}


    # train one batch
    def __fit(self, x1a, x2a, x1b, x2b, weight):
        feed_dict = self.__get_feed_dict(x1a, x2a, x1b, x2b, weight)
        loss,loss_1st,loss_2nd1,loss_2nd2,loss_reg, _ = \
            self.session.run((
                self.loss, 
                self.loss_1st, 
                self.loss_2nd1,
                self.loss_2nd2, 
                self.loss_reg,
                self.optimizer ), feed_dict = feed_dict)
        return loss,loss_1st,loss_2nd1,loss_2nd2,loss_reg


    def __get_learning_rate(self):
        if self.learning_decay:
            return self.session.run(self.learning_rate)
        else:
            return self.learning_rate

    def get_loss(self, x1a, x2a, x1b, x2b, weight):
        feed_dict = self.__get_feed_dict(x1a, x2a, x1b, x2b, weight)
        return self.session.run((
                self.loss,
                self.loss_1st,
                self.loss_2nd1,
                self.loss_2nd2,
                self.loss_reg), feed_dict = feed_dict)

    def get_embedding(self, x1a, x2a):
        return self.session.run(self.ha, feed_dict = {self.x1a:x1a,self.x2a:x2a})

    def get_reconstruct(self, x1a, x2a):
        return self.session.run(self.xa_reconstruct, feed_dict = {self.x1a:x1a,self.x2a:x2a})
    
    def get_w(self):
        return self.session.run((self.w1,self.w2))
        
    def get_b(self):
        return self.session.run((self.b1,self.b2))
        
    def close(self):
        self.session.close()

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)
        self.init = True
