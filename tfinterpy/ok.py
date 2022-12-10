from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.variogram import NestVariogram
from tfinterpy.settings import dtype
from tfinterpy.krigeBase import KrigeBase


class OK(KrigeBase):
    '''
    Ordinary Kriging interpolator.
    '''

    def __init__(self, samples, mode="2d"):
        '''
        See the base class KrigeBase annotation for details.
        '''
        super(OK, self).__init__(samples, mode)

    def execute(self, points, N=8, variogram=None, workerNum=1):
        '''
        See the base class KrigeBase annotation for details.

        :param workerNum: By default, one process is used, and multi-process computation is used when wokerNum>1.
        '''
        if workerNum > 1:
            return self.__multiWorker__(points, N=N, variogram=variogram, workerNum=workerNum)

        isNest = variogram.__class__ == NestVariogram
        self.__calcInnerVars__(variogram, isNest)

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=N, eps=0.0)
        properties = np.zeros((len(points)), dtype=dtype)
        sigmas = np.zeros((len(points)), dtype=dtype)
        kmat = np.ones((N + 1, N + 1), dtype=dtype)
        kmat[N, N] = 0.0
        mvec = np.ones((N + 1,), dtype=dtype)
        for idx, indice in enumerate(nbIdx):
            kmat[:N, :N] = self.innerVars[indice][:, indice]
            if isNest:
                vecs = self.samples[indice, :self._i] - points[idx]
                mvec[:N] = variogram(vecs)
            elif not variogram is None:
                mvec[:N] = variogram(nbd[idx])
            else:
                mvec[:N] = nbd[idx]
            # np.fill_diagonal(kmat,0.0)
            try:
                lambvec = np.linalg.inv(kmat).dot(mvec)
            except:
                lambvec = np.linalg.pinv(kmat).dot(mvec)
            properties[idx] = np.dot(self.samples[indice, self._i], lambvec[:N])
            sigmas[idx] = np.dot(lambvec, mvec)
        return properties, sigmas

    def crossValidate(self, N=8, variogram=None):
        '''
        Perform leave-one-out cross validation on sample points.

        :param N: integer, neighborhood size.
        :param variogram: variogram function or nest variogram object, default None.
            A linear variogram (lambda x:x) is used when the variogram is None.
        :return: tuple, tuple containing absolute mean error, absolute standard deviation error and origin error(ndarray).
        '''
        isNest = variogram.__class__ == NestVariogram
        self.__calcInnerVars__(variogram, isNest)

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        properties = np.zeros((len(self.samples)), dtype=dtype)
        kmat = np.ones((N + 1, N + 1), dtype=dtype)
        kmat[N, N] = 0.0
        mvec = np.ones((N + 1,), dtype=dtype)
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            kmat[:N, :N] = self.innerVars[indice][:, indice]
            if isNest:
                vecs = self.samples[indice, :self._i] - self.samples[idx, :self._i]
                mvec[:N] = variogram(vecs)
            elif not variogram is None:
                mvec[:N] = variogram(nbd[idx][1:])
            else:
                mvec[:N] = nbd[idx][1:]
            # np.fill_diagonal(kmat,0.0)
            try:
                lambvec = np.linalg.inv(kmat).dot(mvec)
            except:
                lambvec = np.linalg.pinv(kmat).dot(mvec)
            properties[idx] = np.dot(self.samples[:, self._i][indice], lambvec[:N])
        error = properties - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error
