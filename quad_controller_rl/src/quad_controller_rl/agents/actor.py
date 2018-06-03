import tensorflow as tf
import numpy as np

class ActorModel():
    def __init__(self,learning_rate=.01,scope="policy"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32,name="state")
            self.target = tf.placeholder(tf.float32,name="target")

            