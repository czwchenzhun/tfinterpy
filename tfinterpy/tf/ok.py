import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tfinterpy.settings import dtype
from tfinterpy.tf.krigeBase import *
from tfinterpy.tf.layers import KMatLayer, MVecLayer, IndiceLayer, BatchConcatenate

def OKModel(innerVars, sampleLocs, samplePros, n=8, variogramLayer=None):
    '''
    Construction a keras model for Ordinary Kriging algorithm.

    :param innerVars: ndarray, the square of the difference between the sampling point properties.
    :param sampleLocs: ndarray, sampling point coordinates.
    :param samplePros: ndarray, sampling point properties.
    :param n: integer, neighborhood size.
    :param variogramLayer: keras' layer, layer representing a variogram function.
    :return: keras' Model object.
    '''
    mat1 = np.ones((n + 1, n + 1),dtype=dtype)
    mat1[n, n] = 0
    mat1 = tf.constant(mat1, dtype=dtype)

    tmp = np.ones((innerVars.shape[0] + 1, innerVars.shape[0] + 1), dtype=dtype)
    tmp[:innerVars.shape[0], :innerVars.shape[0]] = innerVars
    innerVars = tmp

    innerVars = tf.convert_to_tensor(innerVars)
    sampleLocs = tf.convert_to_tensor(sampleLocs)
    samplePros = tf.convert_to_tensor(samplePros)

    indices = layers.Input(shape=(n,), dtype="int32", name="indices")
    points = layers.Input(shape=(sampleLocs.shape[1],), dtype=dtype, name="points")

    locs = IndiceLayer(sampleLocs)(indices)
    mvec = layers.Subtract()([locs, points])
    pro = IndiceLayer(samplePros)(indices)

    tmp2 = tf.zeros((1,),dtype="int32")
    tmp2 = tmp2 + (innerVars.shape[0]-1)

    indices_ = BatchConcatenate()(indices, tmp2)
    kmat = KMatLayer(innerVars)(indices_)
    kmat = kmat * mat1

    if variogramLayer != None:
        mvec = variogramLayer(mvec)
        mvec = BatchConcatenate()(mvec, tf.ones((1, 1), dtype=dtype))
    else:
        mvec = tf.linalg.norm(mvec, axis=2)
        mvec = BatchConcatenate()(mvec, tf.ones((1,), dtype=dtype))
        mvec = layers.Reshape((n + 1, 1))(mvec)
    kmatInv = tf.linalg.pinv(kmat)
    lambvec = layers.Dot(1)([kmatInv, mvec])
    estimate = layers.Dot(1)([pro, lambvec[:, :n]])
    lambvec = layers.Reshape((n + 1,))(lambvec)
    mvec = layers.Reshape((n + 1,))(mvec)
    sigma = layers.Dot(1)([lambvec, mvec])
    model = Model(inputs=[indices, points], outputs=[estimate, sigma])
    return model

class TFOK(TFKrigeBase):
    '''
    Tensorflow version of Simple Kriging interpolator.
    '''

    def __init__(self, samples, mode='2d'):
        '''
        See the base class KrigeBase annotation for details.
        '''
        super(TFOK, self).__init__(samples, mode)

    def modelFunc(self):
        return OKModel