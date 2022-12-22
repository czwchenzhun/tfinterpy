import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tfinterpy.settings import dtype
from tfinterpy.tf.layers import KMatLayer, MVecLayer, IndiceLayer
from tfinterpy.tf.krigeBase import TFKrigeBase


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


class TrendFunc(layers.Layer):
    def __init__(self):
        super(TrendFunc, self).__init__()
        self.items = []

    @tf.function(jit_compile=True)
    def call(self, loc):
        res = tf.concat([[]], 0)
        for f in self.items:
            res = tf.concat([res, [f(loc)]], 0)
        return res

    def __len__(self):
        return len(self.items)


class Linear2(TrendFunc):
    '''
    x+y
    m(x) = a1*x + a2*y
    '''

    def __init__(self):
        super(Linear2, self).__init__()
        self.items.append(lambda p: p[0])
        self.items.append(lambda p: p[1])


class Linear3(TrendFunc):
    '''
    x+y+z
    m(x) = a1*x + a2*y + a3*z
    '''

    def __init__(self):
        super(Linear3, self).__init__()
        self.items.append(lambda p: p[0])
        self.items.append(lambda p: p[1])
        self.items.append(lambda p: p[2])


class Quadratic2(TrendFunc):
    '''
    (x+y)^2
    m(x) = a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y
    '''

    def __init__(self):
        super(Quadratic2, self).__init__()
        # self.items.append(lambda p: p[0])
        # self.items.append(lambda p: p[1])
        # self.items.append(lambda p: p[0] * p[0])
        # self.items.append(lambda p: p[1] * p[1])
        # self.items.append(lambda p: 2 * p[0] * p[1])
        self.items.append(lambda p: 0.5*(3*(p[0]*p[0] + p[1]*p[1] + 2*p[0]+p[1])-1))


class Quadratic3(TrendFunc):
    '''
    (x+y+z)^2
    m(x) = x^2 + y^2 + z^2 + 2*x*y + 2*x*z + 2*y*z
    '''

    def __init__(self):
        super(Quadratic3, self).__init__()
        self.items.append(lambda p: p[0] * p[0])
        self.items.append(lambda p: p[1] * p[1])
        self.items.append(lambda p: p[2] * p[2])
        self.items.append(lambda p: p[0] * p[1])
        self.items.append(lambda p: 2 * p[0] * p[2])
        self.items.append(lambda p: 2 * p[1] * p[2])


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
        trendFunc = Linear2() if dim == 2 else Linear3()

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
    estimate = estimate + layers.Dot(1)([lambvec[:, n:], mvec[:, n:]])
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
