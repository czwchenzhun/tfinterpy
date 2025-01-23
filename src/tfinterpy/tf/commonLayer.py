import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

class InvLayer(layers.Layer):
    def __init__(self):
        super(InvLayer, self).__init__()

    def call(self, mat):
        return tf.linalg.pinv(mat)
    
class MeanLayer(layers.Layer):
    def __init__(self):
        super(MeanLayer, self).__init__()
        
    def call(self, vec):
        return K.mean(vec, axis=1)
    
class SumLayer(layers.Layer):
    def __init__(self):
        super(SumLayer, self).__init__()
        
    def call(self, vec):
        return K.sum(vec, axis=1)