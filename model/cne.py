# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy,time
from scipy import io
from sklearn.metrics import pairwise_distances
from .rbm import RBM



class CNE(object):
    def __init__(self,
                padding              = 'SAME',
                input_dim            = 10000,
                kernel_size          = 3,
                stride               = 2,
                cnndepth             = 2,
                mlpstruct            = [128],
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
                embedding_path       = "embedding"):
        self.padding            = padding
        self.input_dim          = input_dim
        self.kernel_size        = kernel_size
        self.stride             = stride
        self.cnndepth           = cnndepth
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
        self.mlpstruct          = mlpstruct
        self.struct             = [0] * ( 2 * self.cnndepth +1 + len(self.mlpstruct) * 2 )
        self.output_dim         = self.mlpstruct[-1]

        # decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
        if self.learning_decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step = self.global_step, 
                decay_steps = learning_decay_steps, 
                decay_rate  = learning_decay_rate)

        self.xa                 = tf.placeholder(tf.float32, [None, self.input_dim], name="xa")
        self.xb                 = tf.placeholder(tf.float32, [None, self.input_dim], name="xb")
        self.weight             = tf.placeholder(tf.float32, [None,], name="weight")


        def default_logger(s):
            print(s)
        if logger == None:
            self.logger = default_logger
        else:
            self.logger = logger
 
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




    def __conv1d(self, x, w):
        return tf.nn.conv1d(x, w, 
                    stride=self.stride, 
                    padding=self.padding)  

    def __conv1dinv(self, x, w, d):
        return tf.contrib.nn.conv1d_transpose(x, w, 
                    output_shape=[self.batch_size,d,1],
                    stride=self.stride, 
                    padding=self.padding)

        
    def __encoder(self, p, active_func):
        p = tf.expand_dims(p, 2)
        self.struct[0] = self.input_dim
        for i in range(self.cnndepth):
            name = "cnn_encoder_%d"%(i)
            # self.w[name] = tf.Variable(tf.random_normal([self.kernel_size, 1, 1]), name="%s_w"%name)
            self.w[name] = tf.get_variable(name="%s_w"%name,shape=[self.kernel_size, 1, 1],initializer=tf.random_normal_initializer())
            p = self.__conv1d(p,self.w[name])
            # self.b[name] = tf.Variable(tf.random_normal([1,int(p.shape[1]),1]), name="%s_b"%name)
            self.b[name] = tf.get_variable(name="%s_b"%name,shape=[1,int(p.shape[1]),1],initializer=tf.random_normal_initializer())
            # p = active_func( p + self.b[name])
            self.struct[i+1] = int(p.shape[1])
        p = tf.squeeze(p)
        for i in range(len(self.mlpstruct)):
            name = "mlp_encoder_%d"%(i)
            if i == 0:
                # self.w[name] = tf.Variable(tf.random_normal([self.struct[self.cnndepth],self.mlpstruct[i]]),name="%s_w"%name)
                # self.b[name] = tf.Variable(tf.random_normal([self.mlpstruct[i]]), name = "%s_b"%name)
                self.w[name] = tf.get_variable(name="%s_w"%name,shape=[self.struct[self.cnndepth],self.mlpstruct[i]],initializer=tf.random_normal_initializer())
                self.b[name] = tf.get_variable(name="%s_b"%name,shape=[self.mlpstruct[i]],initializer=tf.random_normal_initializer())
            else:
                # self.w[name] = tf.Variable(tf.random_normal([self.mlpstruct[i-1],self.mlpstruct[i]]),name="%s_w"%name)
                # self.b[name] = tf.Variable(tf.random_normal([self.mlpstruct[i]]), name = "%s_b"%name)
                self.w[name] = tf.get_variable(name="%s_w"%name,shape=[self.mlpstruct[i-1],self.mlpstruct[i]],initializer=tf.random_normal_initializer())
                self.b[name] = tf.get_variable(name="%s_b"%name,shape=[self.mlpstruct[i]],initializer=tf.random_normal_initializer())
            p = active_func(tf.matmul(p,self.w[name])+self.b[name])
            self.struct[ i+1+self.cnndepth ] = int(p.shape[1])
        return p
    
    def __decoder(self, p, active_func):
        mlpdepth    = len(self.mlpstruct)
        input_dim   = self.struct[self.cnndepth+1]
        output_dim  = self.struct[self.cnndepth]
        for i in range(mlpdepth):
            name = "mlp_decoder_%d"%(i)
            if i == mlpdepth-1:
                # self.w[name] = tf.Variable(tf.random_normal([input_dim,output_dim]),name="%s_w"%name)
                # self.b[name] = tf.Variable(tf.random_normal([output_dim]), name = "%s_b"%name)
                self.w[name] = tf.get_variable(name="%s_w"%name,shape=[input_dim,output_dim],initializer=tf.random_normal_initializer())
                self.b[name] = tf.get_variable(name="%s_b"%name,shape=[output_dim],initializer=tf.random_normal_initializer())
            else:
                # self.w[name] = tf.Variable(tf.random_normal([self.mlpstruct[mlpdepth-1-i],self.mlpstruct[mlpdepth-2-i]]),name="%s_w"%name)
                # self.b[name] = tf.Variable(tf.random_normal([self.mlpstruct[mlpdepth-2-i]]), name = "%s_b"%name)
                self.w[name] = tf.get_variable(name="%s_w"%name,shape=[self.mlpstruct[mlpdepth-1-i],self.mlpstruct[mlpdepth-2-i]],initializer=tf.random_normal_initializer())
                self.b[name] = tf.get_variable(name="%s_b"%name,shape=[self.mlpstruct[mlpdepth-2-i]],initializer=tf.random_normal_initializer())
            p = active_func(tf.matmul(p,self.w[name])+self.b[name])
            self.struct[ i+1+self.cnndepth+mlpdepth ] = int(p.shape[1])

        p = tf.expand_dims(p, 2)
        for i in range(self.cnndepth):
            name = "cnn_decoder_%d"%(i)
            # self.w[name] = tf.Variable(tf.random_normal([self.kernel_size, 1, 1]), name="%s_w"%name)
            self.w[name] = tf.get_variable(name="%s_w"%name,shape=[self.kernel_size, 1, 1],initializer=tf.random_normal_initializer())
            p = self.__conv1dinv(p,self.w[name], self.struct[self.cnndepth-1-i])
            # self.b[name] = tf.Variable(tf.random_normal([1,int(p.shape[1]),1]), name="%s_b"%name)
            self.b[name] = tf.get_variable(name="%s_b"%name,shape=[1,int(p.shape[1]),1],initializer=tf.random_normal_initializer())
            # p = active_func( p + self.b[name])
            self.struct[i+1+self.cnndepth+2*mlpdepth] = int(p.shape[1])
        p = tf.squeeze(p)
        return p

    def __make_compute_graph(self, active_func = tf.nn.sigmoid):
        self.ha = self.__encoder(self.xa,active_func)
        self.xa_reconstruct = self.__decoder(self.ha,active_func)
        print(self.ha.shape,self.xa_reconstruct.shape)
        tf.get_variable_scope().reuse_variables()
        self.hb = self.__encoder(self.xb,active_func)
        self.xb_reconstruct = self.__decoder(self.hb,active_func)
        print(self.hb.shape,self.xb_reconstruct.shape)
        self.logger("Model Struct: "+str(self.struct))

    
    def __cal_loss(self):
        self.loss_1st = tf.reduce_sum(
                self.weight * tf.reduce_sum(tf.square((self.ha-self.hb)),keep_dims=True))
        self.loss_2nd = \
                tf.reduce_sum(tf.square((self.xa_reconstruct-self.xa)*self.Ba)) + \
                tf.reduce_sum(tf.square((self.xb_reconstruct-self.xb)*self.Bb))
        
        self.loss = self.alpha1 * self.loss_1st + self.alpha2 * self.loss_2nd
        self.loss_reg = None
        for wi in self.w:
            if "encoder" in wi:
                if self.loss_reg == None:
                    self.loss_reg = self.nu1 * tf.nn.l2_loss(self.w[wi]) + self.nu1 * tf.nn.l2_loss(self.b[wi])
                else:
                    self.loss_reg = tf.add(self.loss_reg, self.nu1 * tf.nn.l2_loss(self.w[wi]) + self.nu1 * tf.nn.l2_loss(self.b[wi]))
            elif "decoder" in wi:
                if self.loss_reg == None:
                    self.loss_reg = self.nu2 * tf.nn.l2_loss(self.w[wi]) + self.nu2 * tf.nn.l2_loss(self.b[wi])
                else:
                    self.loss_reg = tf.add(self.loss_reg, self.nu2 * tf.nn.l2_loss(self.w[wi]) + self.nu2 * tf.nn.l2_loss(self.b[wi]))
        self.loss += self.loss_reg
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
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
        loss,loss_1st,loss_2nd,loss_reg = 0,0,0,0
        # for xa,xb,weight,indexa,indexb in sample_fun(graph,batch_size):
        #     t_loss,t_loss_1st,t_loss_2nd,loss_reg = self.get_loss(xa, xb, weight)
        #     loss += t_loss
        #     loss_1st += t_loss_1st
        #     loss_2nd += t_loss_2nd
        # self.logger("Epoch %3d Loss(All=%5.1f 1st=%5.1f 2nd=%5.1f reg=%5.1f)"%(
        #     current_epoch, loss,loss_1st,loss_2nd,loss_reg))
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
                for xa,xb,weight,indexa,indexb in sample_fun(graph,batch_size):
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

    def get_embedding(self, X):
        return self.session.run(self.ha, feed_dict = {self.xa:X})

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