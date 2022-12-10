import tensorflow as tf
from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.tf.variogramLayer import NestVariogramLayer
from tfinterpy.settings import dtype
from tfinterpy import krigeBase
import warnings

tf.keras.backend.set_floatx(dtype)


class TFKrigeBase(krigeBase.KrigeBase):
    '''
    Tensorflow version of Krige Interpolator Base.
    '''

    def __init__(self, samples, mode='2d'):
        '''
        See the base class KrigeBase annotation for details.
        '''
        super(TFKrigeBase, self).__init__(samples, mode)

    def execute(self, points, N=8, variogram=None, batch_size=10000, batch_num=3, workerNum=1, device='/CPU:0',
                **kwargs):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param variogram: keras' layer, layer representing a variogram function.
        :param batch_size: integer, size of each batch of data to be calculated.
        :param batch_num: integer, batch_size * batch_num number of points will find their nearest neighbors and then feed them to TensorFlow graph.
        :param workerNum: By default, one process is used, and multi-process computation is used when wokerNum>1.
            Notice! If GPU device is specified, the multi-process cannot be enabled.
        :param device: Specified computing device, default value is '/CPU:0'.
        :return: tuple, tuple containing tow ndarray.
            The first ndarray representing the interpolation result,
            the second ndarray representing the kriging variance.
        '''
        if workerNum > 1:
            warnings.warn("Multi-process tasks are not recommended when the data volume is small")
            if 'GPU' in device.upper():
                warnings.warn(
                    "It is not recommended to use the GPU for multi-process tasks, and the speed may actually decrease!")
            # assert 'GPU' not in device.upper(), "multi-process with GPU is not supported!"
            return self.__multiWorker__(points, N=N, variogram=variogram, batch_size=batch_size, workerNum=workerNum,
                                        batch_num=batch_num,
                                        device=device, **kwargs)

        isNest = variogram.__class__ == NestVariogramLayer
        self.__calcInnerVars__(variogram, isNest)
        self.model = self.buildModel(self.innerVars, self.samples[:, :self._i], self.samples[:, self._i], N, variogram,
                                     **kwargs)
        with tf.device(device):
            tree = cKDTree(self.samples[:, :self._i])
            step = batch_size * batch_num
            num = int(np.ceil(len(points) / step))
            pros = np.empty((0, 1), dtype=np.float32)
            sigmas = np.empty((0, 1), dtype=np.float32)
            for i in range(num):
                begin = i * step
                end = (i + 1) * step
                if end > len(points):
                    end = len(points)
                points_ = points[begin:end]
                _, nbIdx = tree.query(points_, k=N, eps=0.0)
                pro, sigma = self.model.predict([nbIdx, points_], batch_size=batch_size)
                pros = np.append(pros, pro)
                sigmas = np.append(sigmas, sigma)
        return pros, sigmas

    def crossValidate(self, N=8, variogram=None, **kwargs):
        '''
        Perform leave-one-out cross validation on sample points.

        :param N: integer, neighborhood size.
        :param variogram: keras' layer, layer representing a variogram function.
        :return: tuple, tuple containing absolute mean error, absolute standard deviation error and origin error(ndarray).
        '''
        isNest = variogram.__class__ == NestVariogramLayer
        self.__calcInnerVars__(variogram, isNest)
        self.model = self.buildModel(self.innerVars, self.samples[:, :self._i], self.samples[:, self._i], N, variogram,
                                     **kwargs)
        tree = cKDTree(self.samples[:, :self._i])
        _, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        pros, _ = self.model.predict([nbIdx[:, 1:], self.samples[:, :self._i]], batch_size=10000)
        pros = pros.reshape(-1)
        error = pros - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error

    def crossValidateKFold(self, K=10, N=8, variogram=None, **kwargs):
        '''
        See the base class annotation for details.
        '''
        return super(TFKrigeBase, self).crossValidateKFold(K, N, variogram, **kwargs)

    def buildModel(self, innerVars, sampleLocs, samplePros, N, variogram, **kwargs):
        '''
        return a TensorFlow Model.
        '''
        return self.modelFunc()(innerVars, sampleLocs, samplePros, N, variogram, **kwargs)

    def modelFunc(self):
        '''
        return a function which return a TensorFlow Model, subclass
        '''
        pass
