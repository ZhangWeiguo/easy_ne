import tensorflow as tf
import numpy as np
'''
RBM 模型
参考文章：https://blog.csdn.net/tsb831211/article/details/52757261
'''

class RBM:
    def __init__(self, shape, learning_rate, batch_size):
        '''
        shape = [input_dim, output_dim]
        learing_rate
        batch_size
        w * v + bh = h
        '''
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sess = tf.Session()
        stddev = 1.0 / np.sqrt(shape[0])

        self.w  = tf.Variable(tf.random_normal([shape[0], shape[1]], stddev = stddev), name = "w")
        self.bh = tf.Variable(tf.zeros(shape[1]), name = "bh")
        self.v  = tf.placeholder(tf.float32, [None, shape[0]])
        self.bv = tf.Variable(tf.zeros(shape[0]), name = "bv")

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.build_model()


    def build_model(self):
        self.h = self.sample(tf.nn.relu(tf.sigmoid(tf.matmul(self.v, self.w) + self.bh)))

        # gibbs sample
        # p(v) = tf.sigmoid(tf.matmul(self.h, tf.transpose(self.w)) + self.bv)
        # p(h) = tf.sigmoid(tf.matmul(v_sample, self.w) + self.bh)
        v_sample = self.sample(tf.sigmoid(tf.matmul(self.h, tf.transpose(self.w)) + self.bv))
        h_sample = self.sample(tf.sigmoid(tf.matmul(v_sample, self.w) + self.bh))

        lr          = self.learning_rate / tf.to_float(self.batch_size)
        w_tmp1      = tf.matmul(tf.transpose(v_sample), h_sample)
        w_tmp0      = tf.matmul(tf.transpose(self.v), self.h)
        w_adder     = self.w.assign_add(lr  * (w_tmp0 - w_tmp1))
        bv_adder    = self.bv.assign_add(lr * tf.reduce_mean(self.v - v_sample, 0))
        bh_adder    = self.bh.assign_add(lr * tf.reduce_mean(self.h - h_sample, 0))
        self.upt    = [w_adder, bv_adder, bh_adder]
        self.error  = tf.reduce_sum(tf.pow(self.v - v_sample, 2))
    
    def fit(self, data):
        _, e = self.sess.run((self.upt, self.error), feed_dict = {self.v : data})
        return e
    
    def sample(self, probs):
        # 是 probs > random 则等于1?
        # 还是 probs + random >= 1 则等于1?
        # 这两者应该是一样的
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
        # return tf.nn.relu(tf.random_uniform(tf.shape(probs),0,1)- probs)
    
    def get_para(self):
        return self.sess.run([self.w, self.bv, self.bh])
    
    def predict(self, data):
        # data.shape = [None, input_dim]
        return self.sess.run(self.h, feed_dict = {self.v : data})
    
    def close(self):
        self.sess.close()