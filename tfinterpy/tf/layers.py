import tensorflow as tf
from tensorflow.keras import layers

class KMatLayer(layers.Layer):
    def __init__(self, innerVars):
        super(KMatLayer, self).__init__()
        self.innerVars = innerVars

    @tf.function(jit_compile=True)
    def call(self, indices):
        return tf.vectorized_map(lambda indice: tf.gather(tf.gather(self.innerVars, indice), indice, axis=1), indices)


class MVecLayer(layers.Layer):
    def __init__(self, sampleLocs):
        super(MVecLayer, self).__init__()
        self.sampleLocs = sampleLocs

    @tf.function(jit_compile=True)
    def call(self, indices, points):
        return tf.vectorized_map(lambda i: tf.gather(self.sampleLocs, indices[i]) - points[i],
                                 tf.range(tf.shape(indices)[0]))


class IndiceLayer(layers.Layer):
    def __init__(self, data):
        super(IndiceLayer, self).__init__()
        self.data = data

    @tf.function(jit_compile=True)
    def call(self, indices):
        return tf.vectorized_map(lambda indice: tf.gather(self.data, indice), indices)


class BatchConcatenate(layers.Layer):
    def __init__(self):
        super(BatchConcatenate, self).__init__()

    @tf.function(jit_compile=True)
    def call(self, x, y, axis=0):
        return tf.vectorized_map(lambda a: tf.concat([a, y], axis), x)