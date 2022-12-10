import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tfinterpy.settings import dtype
from tfinterpy.tf.krigeBase import TFKrigeBase
from tfinterpy.tf.layers import KMatLayer, MVecLayer, IndiceLayer


def SKModel(innerVars, sampleLocs, samplePros, n=8, variogramLayer=None):
    '''
    Construction a keras model for Simple Kriging algorithm.

    :param innerVars: ndarray, the square of the difference between the sampling point properties.
    :param sampleLocs: ndarray, sampling point coordinates.
    :param samplePros: ndarray, sampling point properties.
    :param n: integer, neighborhood size.
    :param variogramLayer: keras' layer, layer representing a variogram function.
    :return: keras' Model object.
    '''
    innerVars = tf.convert_to_tensor(innerVars)
    sampleLocs = tf.convert_to_tensor(sampleLocs)
    samplePros = tf.convert_to_tensor(samplePros)
    indices = layers.Input(shape=(n,), dtype=tf.int32, name="indices")
    points = layers.Input(shape=(sampleLocs.shape[1],), dtype=dtype, name="points")
    kmat = KMatLayer(innerVars)(indices)
    mvec = MVecLayer(sampleLocs)(indices, points)
    pro = IndiceLayer(samplePros)(indices)
    if variogramLayer != None:
        mvec = variogramLayer(mvec)
    else:
        mvec = tf.linalg.norm(mvec, axis=2)
        mvec = layers.Reshape((n, 1))(mvec)
    kmatInv = tf.linalg.pinv(kmat)
    lambvec = layers.Dot(1)([kmatInv, mvec])
    estimate = layers.Dot(1)([pro, lambvec])
    promean = K.mean(pro, axis=1)
    eps = K.sum(lambvec, axis=1)
    promean = layers.Reshape((1,))(promean)
    eps = layers.Reshape((1,))(eps)
    estimate = estimate + promean * (1 - eps)
    lambvec = layers.Reshape((n,))(lambvec)
    mvec = layers.Reshape((n,))(mvec)
    sigma = layers.Dot(1)([lambvec, mvec])
    model = Model(inputs=[indices, points], outputs=[estimate, sigma])
    return model


class TFSK(TFKrigeBase):
    '''
    Tensorflow version of Simple Kriging interpolator.
    '''

    def __init__(self, samples, mode='2d'):
        '''
        See the base class KrigeBase annotation for details.
        '''
        super(TFSK, self).__init__(samples, mode)

    def modelFunc(self):
        return SKModel
