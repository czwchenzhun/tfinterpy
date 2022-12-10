from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.variogram import NestVariogram
from tfinterpy.settings import dtype
from tfinterpy.krigeBase import KrigeBase


class SK(KrigeBase):
    '''
    Simple Kriging interpolator.
    '''

    def __init__(self, samples, mode="2d"):
        '''
        See the base class KrigeBase annotation for details.
        '''
        super(SK, self).__init__(samples, mode)

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
        for idx, indice in enumerate(nbIdx):
            kmat = self.innerVars[indice][:, indice]
            if isNest:
                mvec = self.samples[indice, :self._i] - points[idx]
                mvec = variogram(mvec)
            elif not variogram is None:
                mvec = variogram(nbd[idx])
            else:
                mvec = nbd[idx]
            try:
                lambvec = np.linalg.inv(kmat).dot(mvec)
            except:
                lambvec = np.linalg.pinv(kmat).dot(mvec)
            pro = np.dot(self.samples[:, self._i][indice], lambvec)
            properties[idx] = pro + np.mean(self.samples[:, self._i][indice]) * (1 - np.sum(lambvec))
            sigmas[idx] = np.dot(lambvec, mvec)
        return properties, sigmas

    def crossValidate(self, N=8, variogram=None):
        '''
        See the base class KrigeBase annotation for details.
        '''
        isNest = variogram.__class__ == NestVariogram
        self.__calcInnerVars__(variogram, isNest)

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        properties = np.zeros((len(self.samples)), dtype=dtype)
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            kmat = self.innerVars[indice][:, indice]
            if isNest:
                mvec = self.samples[indice, :self._i] - self.samples[idx, :self._i]
                mvec = variogram(mvec)
            elif not variogram is None:
                mvec = variogram(nbd[idx][1:])
            else:
                mvec = nbd[idx][1:]
            try:
                lambvec = np.linalg.inv(kmat).dot(mvec)
            except:
                lambvec = np.linalg.pinv(kmat).dot(mvec)
            pro = np.dot(self.samples[:, self._i][indice], lambvec)
            properties[idx] = pro + np.mean(self.samples[:, self._i][indice]) * (1 - np.sum(lambvec))
        error = properties - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error
