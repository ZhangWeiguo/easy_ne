# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy,time
from scipy import io
from sklearn.metrics import pairwise_distances
from .rbm import RBM

class SDNE(object):
    def __init__(self,
                struct               = [100,2], 
                alpha1               = 1e-1,
                alpha2               = 1,
                beta                 = 5.0,
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
                embedding_path       = "embedding",
                sparse               = True,
                sparse_alpha         = 10):
        self.struct             = struct
        self.layers             = len(self.struct)
        self.alpha1             = alpha1
        self.alpha2             = alpha2
        self.beta               = beta
        self.nu1                = nu1
        self.nu2                = nu2
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.min_re_err         = min_re_err
        self.learning_rate      = learning_rate
        self.learning_decay     = learning_decay
        self.init               = False
        self.sparse             = sparse
        self.sparse_alpha       = sparse_alpha

        # decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
        if self.learning_decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step = self.global_step, 
                decay_steps = learning_decay_steps, 
                decay_rate  = learning_decay_rate)

        self.xa                 = tf.placeholder(tf.float32, [None, self.struct[0]], name="xa")
        self.xb                 = tf.placeholder(tf.float32, [None, self.struct[0]], name="xb")
        self.weight             = tf.placeholder(tf.float32, [None,], name="weight")


        def default_logger(s):
            print(s)
 
        self.Ba = tf.add(tf.ones_like(self.xa),tf.sign(self.xa)*(self.beta-1), name="Ba")
        self.Bb = tf.add(tf.ones_like(self.xb),tf.sign(self.xb)*(self.beta-1), name="Bb")


        self.ha                 = None
        self.hb                 = None
        self.xa_reconstruct     = None
        self.xb_reconstruct     = None
        self.w                  = {}
        self.b                  = {}
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
        self.Mid = []
        def encoder(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                X = active_func(tf.matmul(X, self.w[name]) + self.b[name])
                # if i == self.layers - 2:
                #     X = active_func(tf.matmul(X, self.w[name]) + self.b[name])
                # else:
                #     X = tf.nn.relu(tf.matmul(X, self.w[name]) + self.b[name])
                self.Mid.append(X)
            return X
        def decoder(X):
            for i in range(self.layers - 1):
                name = "decoder" + str(i)
                X = active_func(tf.matmul(X, self.w[name]) + self.b[name])
                # if i == 0:
                #     X = active_func(tf.matmul(X, self.w[name]) + self.b[name])
                # else:
                #     X = tf.nn.relu(tf.matmul(X, self.w[name]) + self.b[name])
                self.Mid.append(X)
            return X
        struct = self.struct
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.w[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = "w"+name)
            self.b[name] = tf.Variable(tf.random_normal([struct[i+1]]), name = "b"+name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.w[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = "w"+name)
            self.b[name] = tf.Variable(tf.random_normal([struct[i+1]]), name = "b"+name)
        struct.reverse()
        self.ha = encoder(self.xa)
        self.xa_reconstruct = decoder(self.ha)
        self.hb = encoder(self.xb)
        self.xb_reconstruct = decoder(self.hb)


    def __cal_spare_loss(self, y):
        p = 0.01
        pj = tf.reduce_mean(tf.abs(y)) + p
        return self.sparse_alpha * tf.reduce_sum(p * tf.log(p / pj) + (1-p) * tf.log((1 - p)/(1 - pj)))


    def __cal_loss(self):
        self.loss_1st = tf.reduce_sum(
                self.weight * tf.reduce_sum(tf.square((self.ha-self.hb)),keep_dims=True))
        self.loss_2nd = \
                tf.reduce_sum(tf.square((self.xa_reconstruct-self.xa)*self.Ba)) + \
                tf.reduce_sum(tf.square((self.xb_reconstruct-self.xb)*self.Bb))
        
        self.loss = self.alpha1 * self.loss_1st + self.alpha2 * self.loss_2nd
        if self.sparse:
            self.loss += self.__cal_spare_loss(self.ha) + self.__cal_spare_loss(self.hb)
        self.loss_reg = None
        for wi in self.w:
            if "encoder" in wi:
                if self.loss_reg == None:
                    self.loss_reg = self.nu1 * tf.nn.l2_loss(self.w[wi]) + self.nu1 * tf.nn.l2_loss(self.b[wi])
                else:
                    self.loss_reg = tf.add(self.loss_reg, self.nu1 * tf.nn.l2_loss(self.w[wi]) + self.nu1 * tf.nn.l2_loss(self.b[wi]))
            elif "decoder" in wi:
                if self.loss_reg == None:
                    self.loss_reg = self.nu2 * tf.nn.l2_loss(self.w[wi])  + self.nu2 * tf.nn.l2_loss(self.b[wi])
                else:
                    self.loss_reg = tf.add(self.loss_reg, self.nu2 * tf.nn.l2_loss(self.w[wi]) + self.nu2 * tf.nn.l2_loss(self.b[wi]))
        self.loss += self.loss_reg
        # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
           self.loss,global_step=self.global_step)

    def rbm_init(self, sample_fun, graph, batch_size):
        self.logger("Init Para By RBM Model Begin")
        shape = self.struct
        rbms = []
        w,b = {},{}
        for i in range(self.layers - 1):
            rbm_unit = RBM(
                shape           = [shape[i], shape[i+1]], 
                batch_size      = self.batch_size, 
                learning_rate   = self.learning_rate)
            rbms.append(rbm_unit)
            for epoch in range(3):
                error = 0
                for xa,xb,_,_,_ in sample_fun(graph,batch_size):
                    mini_batch = xa
                    error_batch = 0
                    for k in range(len(rbms)-1):
                        mini_batch = rbms[k].predict(mini_batch)
                    error_batch += rbm_unit.fit(mini_batch)
                    mini_batch = xb
                    for k in range(len(rbms)-1):
                        mini_batch = rbms[k].predict(mini_batch)
                    error_batch += rbm_unit.fit(mini_batch)
                    error += error_batch
                self.logger("%d Layer: Rbm Epochs %3d Error: %5.6f"%(len(rbms),epoch,error))
            W, bv, bh = rbm_unit.get_para()
            name = "encoder" + str(i)
            w[name] = W
            b[name] = bh
            name = "decoder" + str(self.layers - i - 2)
            w[name] = W.transpose()
            b[name] = bv
        self.init = True
        self.session.run(tf.global_variables_initializer())
        for key in w:
            self.w[key].assign(w[key])
            self.session.run(self.w[key])
        for key in b:
            self.b[key].assign(b[key])
            self.session.run(self.b[key])
        self.logger("Init Para By RBM Model Done")

    # train all data by batch
    def train(self, sample_fun, graph, batch_size):
        if not self.init:
            self.session.run(tf.global_variables_initializer())
            self.rbm_init(sample_fun, graph, batch_size)
        self.tensor_writer = tf.summary.FileWriter("TensorGraph.log", tf.get_default_graph())
        batch_size       = self.batch_size
        model_path       = self.model_path
        embedding_path   = self.embedding_path
        current_epoch    = 0
        last_loss        = 0
        loss,loss_1st,loss_2nd,loss_reg = 0,0,0,0
        for k,(xa,xb,weight,indexa,indexb) in enumerate(sample_fun(graph,batch_size)):
            if k >= 100:
                break
            t_loss,t_loss_1st,t_loss_2nd,loss_reg = self.get_loss(xa, xb, weight)
            loss += t_loss
            loss_1st += t_loss_1st
            loss_2nd += t_loss_2nd
        self.logger("Epoch %3d Loss(All=%5.1f 1st=%5.1f 2nd=%5.1f reg=%5.1f)"%(
            current_epoch, loss,loss_1st,loss_2nd,loss_reg))
        while True:
            if current_epoch < self.epochs:
                current_epoch += 1
                embedding = None
                node_number = len(graph.edges)
                start = 0
                for xa,xb,weight,indexa,indexb in sample_fun(graph,batch_size):
                    t0 = time.time()
                    loss,loss_1st,loss_2nd,loss_reg = self.__fit(xa,xb,weight)
                    train_time = time.time() - t0
                    learning_rate = self.__get_learning_rate()
                    self.logger("Epoch(%3d) LR(%1.5f) Time(Train=%1.5f) Rate(%5d/%5d) Loss(All=%5.1f 1st=%5.1f 2nd=%5.1f reg=%5.1f)"%(
                        current_epoch, learning_rate, train_time,start, node_number, loss,loss_1st,loss_2nd,loss_reg))
                    start += batch_size
                model_path_epoch = model_path + ".%s"%str(current_epoch)
                embedding_path_epoch = embedding_path + ".%s"%str(current_epoch)
                # self.save_model(model_path_epoch)
                self.save_model(model_path)

                loss,loss_1st,loss_2nd,loss_reg = 0,0,0,0
                embedding = self.get_embedding(graph.adjacent_matrix.todense())
                for k,(xa,xb,weight,indexa,indexb) in enumerate(sample_fun(graph,batch_size)):
                    if k >= 100:
                        break
                    t_loss,t_loss_1st,t_loss_2nd,loss_reg = self.get_loss(xa,xb,weight)
                    loss += t_loss
                    loss_1st += t_loss_1st
                    loss_2nd += t_loss_2nd
                # io.savemat(embedding_path_epoch, {"embedding":embedding})
                io.savemat(embedding_path,{"embedding":embedding})
                self.logger("Epoch %3d Loss(All=%5.1f 1st=%5.1f 2nd=%5.1f reg=%5.1f)"%(
                    current_epoch, loss,loss_1st,loss_2nd,loss_reg))
                if last_loss > 0 and (last_loss - loss)/last_loss <= self.min_re_err and loss < last_loss:
                    self.logger("Stop Iter As Loss Reduce Smaller Than MinReError")
                    return
            else:
                break
        self.tensor_writer.close()



    def train_sub(self, sample_fun, graph, batch_size, nodes):
        if not self.init:
            self.session.run(tf.global_variables_initializer())
            # self.rbm_init(sample_fun, graph, batch_size)
        self.tensor_writer = tf.summary.FileWriter("TensorGraph.log", tf.get_default_graph())
        batch_size       = self.batch_size
        model_path       = self.model_path
        embedding_path   = self.embedding_path
        current_epoch    = 0
        last_loss        = 0
        loss,loss_1st,loss_2nd,loss_reg = 0,0,0,0
        for k,(xa,xb,weight,indexa,indexb) in enumerate(sample_fun(graph,batch_size,nodes)):
            if k >= 100:
                break
            t_loss,t_loss_1st,t_loss_2nd,loss_reg = self.get_loss(xa, xb, weight)
            loss += t_loss
            loss_1st += t_loss_1st
            loss_2nd += t_loss_2nd
        self.logger("Epoch %3d Loss(All=%5.1f 1st=%5.1f 2nd=%5.1f reg=%5.1f)"%(
            current_epoch, loss,loss_1st,loss_2nd,loss_reg))
        while True:
            if current_epoch < self.epochs:
                current_epoch += 1
                embedding = None
                node_number = len(graph.edges)
                start = 0
                for xa,xb,weight,indexa,indexb in sample_fun(graph,batch_size,nodes):
                    t0 = time.time()
                    loss,loss_1st,loss_2nd,loss_reg = self.__fit(xa,xb,weight)
                    train_time = time.time() - t0
                    learning_rate = self.__get_learning_rate()
                    self.logger("Epoch(%3d) LR(%1.5f) Time(Train=%1.5f) Rate(%5d/%5d) Loss(All=%5.1f 1st=%5.1f 2nd=%5.1f reg=%5.1f)"%(
                        current_epoch, learning_rate, train_time,start, node_number, loss,loss_1st,loss_2nd,loss_reg))
                    start += batch_size
                model_path_epoch = model_path + ".%s"%str(current_epoch)
                embedding_path_epoch = embedding_path + ".%s"%str(current_epoch)
                # self.save_model(model_path_epoch)
                self.save_model(model_path)

                loss,loss_1st,loss_2nd,loss_reg = 0,0,0,0
                embedding = self.get_embedding(graph.adjacent_matrix.todense()[:,nodes])
                for k,(xa,xb,weight,indexa,indexb) in enumerate(sample_fun(graph,batch_size,nodes)):
                    if k >= 100:
                        break
                    t_loss,t_loss_1st,t_loss_2nd,loss_reg = self.get_loss(xa,xb,weight)
                    loss += t_loss
                    loss_1st += t_loss_1st
                    loss_2nd += t_loss_2nd
                # io.savemat(embedding_path_epoch, {"embedding":embedding})
                io.savemat(embedding_path,{"embedding":embedding})
                self.logger("Epoch %3d Loss(All=%5.1f 1st=%5.1f 2nd=%5.1f reg=%5.1f)"%(
                    current_epoch, loss,loss_1st,loss_2nd,loss_reg))
                if last_loss > 0 and (last_loss - loss)/last_loss <= self.min_re_err and loss < last_loss:
                    self.logger("Stop Iter As Loss Reduce Smaller Than MinReError")
                    return
            else:
                break
        self.tensor_writer.close()

    def __get_feed_dict(self, xa, xb, weight):
        return {self.xa     : xa,
                self.xb     : xb,
                self.weight : weight}


    # train one batch
    def __fit(self, xa, xb, weight):
        feed_dict = self.__get_feed_dict(xa, xb, weight)
        loss,loss_1st,loss_2nd,loss_reg, _ = \
            self.session.run((
                self.loss, 
                self.loss_1st, 
                self.loss_2nd, 
                self.loss_reg,
                self.optimizer ), feed_dict = feed_dict)
        return loss,loss_1st,loss_2nd,loss_reg

    def __get_learning_rate(self):
        if self.learning_decay:
            return self.session.run(self.learning_rate)
        else:
            return self.learning_rate

    def get_loss(self, xa, xb, weight):
        feed_dict = self.__get_feed_dict(xa, xb, weight)
        return self.session.run((
                self.loss,self.loss_1st,
                self.loss_2nd,self.loss_reg), 
                feed_dict = feed_dict)

    def get_embedding(self, X, batch=5000):
        if isinstance(X,type(numpy.array([1,2]))):
            sparse = False
        else:
            sparse = True
        N = X.shape[0]
        if sparse:
            start = 0
            while start < N:
                end = min(start+batch,N)
                if start == 0:
                    embedding = self.session.run(
                        self.ha, 
                        feed_dict = {self.xa : X[start:end].todense()})
                else:
                    embedding = numpy.vstack([
                        embedding, 
                        self.session.run(
                            self.ha, 
                            feed_dict = {self.xa : X[start:end].todense()})
                    ])
                start = end
        else:
            start = 0
            while start < N:
                end = min(start+batch,N)
                if start == 0:
                    embedding = self.session.run(
                        self.ha, 
                        feed_dict = {self.xa : X[start:end]})
                else:
                    embedding = numpy.vstack([
                        embedding, 
                        self.session.run(
                            self.ha, 
                            feed_dict = {self.xa : X[start:end]})
                    ])
                start = end      
        return embedding

    def get_reconstruct(self, X):
        return self.session.run(self.xa_reconstruct, feed_dict = {self.xa:X})
    
    def get_w(self):
        return self.session.run(self.w)
        
    def get_b(self):
        return self.session.run(self.b)
        
    def close(self):
        self.session.close()

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)
        self.init = True