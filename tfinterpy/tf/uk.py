import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tfinterpy.settings import dtype
from tfinterpy.tf.layers import KMatLayer, MVecLayer, IndiceLayer
from tfinterpy.tf.krigeBase import TFKrigeBase
from tfinterpy.tf.ukTrendFunc import *


class ConcatMVecLayer(layers.Layer):
    '''
    Used to add [1, x, y, z] to the end of the variogram
    vector on the right side of the equation.
    '''

    def __init__(self):
        super(ConcatMVecLayer, self).__init__()

    @tf.function(jit_compile=True)
    def call(self, mvec, points, trendFunc):
        one = tf.constant([1], dtype=dtype)
        # return tf.map_fn(lambda x:tf.concat([x[0], one, x[1]], 0), (mvec, points), fn_output_signature=dtype)
        return tf.map_fn(lambda x: tf.concat([x[0], one, trendFunc(x[1])], 0), (mvec, points),
                         fn_output_signature=dtype)


class ConcatKMatLayer(layers.Layer):
    def __init__(self, sampleLocs):
        super(ConcatKMatLayer, self).__init__()
        self.sampleLocs = sampleLocs

    @tf.function(jit_compile=True)
    def call(self, kmat, indices, trendFunc):
        n = indices.shape[1]
        dim = self.sampleLocs.shape[1]

        def func(x):
            kmat, indice = x
            ones = tf.ones((n, 1), dtype=dtype)
            locs = tf.gather(self.sampleLocs, indice)
            items = tf.map_fn(lambda loc: trendFunc(loc), locs, fn_output_signature=dtype)
            ones_items = tf.concat([ones, items], axis=1)
            kmat = tf.concat([kmat, ones_items], axis=1)
            zeros = tf.zeros((len(trendFunc) + 1, len(trendFunc) + 1), dtype=dtype)
            addon = tf.concat([tf.transpose(ones_items), zeros], axis=1)
            return tf.concat([kmat, addon], axis=0)

        return tf.map_fn(func, (kmat, indices), fn_output_signature=dtype)

def UKModel(innerVars, sampleLocs, samplePros, n=8, variogramLayer=None, trendFunc=None):
    '''
    Construction a keras model for Ordinary Kriging algorithm.

    :param innerVars: ndarray, the square of the difference between the sampling point properties.
    :param sampleLocs: ndarray, sampling point coordinates.
    :param samplePros: ndarray, sampling point properties.
    :param n: integer, neighborhood size.
    :param variogramLayer: keras' layer, layer representing a variogram function.
    :return: keras' Model object.
    '''
    dim = sampleLocs.shape[1]
    if trendFunc == None:
        trendFunc = Linear2D() if dim == 2 else Linear3D()

    innerVars = tf.convert_to_tensor(innerVars)
    sampleLocs = tf.convert_to_tensor(sampleLocs)
    samplePros = tf.convert_to_tensor(samplePros)

    indices = layers.Input(shape=(n,), dtype="int32", name="indices")
    points = layers.Input(shape=(dim,), dtype=dtype, name="points")

    kmat = KMatLayer(innerVars)(indices)
    kmat = ConcatKMatLayer(sampleLocs)(kmat, indices, trendFunc)

    mvec = MVecLayer(sampleLocs)(indices, points)
    if variogramLayer != None:
        mvec = variogramLayer(mvec)
        mvec = layers.Reshape((n,))(mvec)
        mvec = ConcatMVecLayer()(mvec, points, trendFunc)
    else:
        mvec = tf.linalg.norm(mvec, axis=2)
        mvec = ConcatMVecLayer()(mvec, points, trendFunc)
    mvec = layers.Reshape((n + len(trendFunc) + 1, 1))(mvec)

    pro = IndiceLayer(samplePros)(indices)

    kmatInv = tf.linalg.pinv(kmat)
    lambvec = layers.Dot(1)([kmatInv, mvec])
    estimate = layers.Dot(1)([pro, lambvec[:, :n]])
    lambvec = layers.Reshape((n + len(trendFunc) + 1,))(lambvec)
    mvec = layers.Reshape((n + len(trendFunc) + 1,))(mvec)
    sigma = layers.Dot(1)([lambvec, mvec])
    # estimate = estimate + layers.Dot(1)([lambvec[:, n:], mvec[:, n:]])#这一行不需要，测试二次趋势函数发现不论是+还是-这一行都会导致插值结果误差极大，大得离谱
    model = Model(inputs=[indices, points], outputs=[estimate, sigma])
    return model


class TFUK(TFKrigeBase):
    '''
    Tensorflow version of Simple Kriging interpolator.
    '''

    def __init__(self, samples, mode='2d'):
        '''
        See the base class KrigeBase annotation for details.
        '''
        super(TFUK, self).__init__(samples, mode)

    def execute(self, points, N=8, variogram=None, batch_size=10000, batch_num=3, workerNum=1, device='/CPU:0',
                trendFunc=None):
        return super(TFUK, self).execute(points, N, variogram, batch_size, batch_num, workerNum, device,
                                         trendFunc=trendFunc)

    def crossValidateKFold(self, K=10, N=8, variogram=None, trendFunc=None):
        return super(TFUK, self).crossValidateKFold(K, N, variogram, trendFunc=trendFunc)

    def crossValidate(self, N=8, variogram=None, trendFunc=None):
        return super(TFUK, self).crossValidate(N, variogram, trendFunc=trendFunc)

    def modelFunc(self):
        return UKModel
