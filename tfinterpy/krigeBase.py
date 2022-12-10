import numpy as np
from tfinterpy.utils import kSplit, calcVecs
from tfinterpy.settings import dtype
from multiprocessing import Pool
from functools import partial

class KrigeBase:
    '''
    Krige Interpolator Base
    '''

    def __init__(self, samples, mode='2d'):
        '''
        Initialize the interpolator using sample points.

        :param samples: ndarray, array containing all sample points. The last column must be the properties.
            For the case of two-dimensional interpolation, where each item is represented by [x,y,property].
            For the case of three-dimensional interpolation, where each item is represented by [x,y,z,property].
        :param mode: str, '2d' or '3d'.
        '''
        if samples.dtype!=dtype:
            self.samples = samples.astype(dtype)
        else:
            self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogram=None, **kwargs):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param variogram: variogram function.
        :return: tuple, tuple containing tow ndarray.
            The first ndarray representing the interpolation result,
            the second ndarray representing the kriging variance.
        '''
        pass

    def crossValidateKFold(self, K=10, N=8, variogram=None, **kwargs):
        '''
        Perform k-fold cross validation on sample points.

        :param K: integer.
        :param N: integer, neighborhood size.
        :param variogram: variogram function.
        :return: tuple, tuple containing three list.
            The first list contains the absolute mean error for each fold,
            the second list contains the absolute standard deviation error for each fold,
            the last list contains the origin error for each fold.
        '''
        splits = kSplit(self.samples, K)
        absErrorMeans = []
        absErrorStds = []
        originalErrorList = []
        for i in range(K):
            concatenateList = []
            for j in range(K):
                if j == i:
                    continue
                concatenateList.append(splits[j])
            p1 = np.concatenate(concatenateList)
            p2 = splits[i]
            if len(p2) == 0:
                break
            exe = self.__class__(p1, self.mode)
            es, _ = exe.execute(p2[:, :self._i], N, variogram, **kwargs)
            error = p2[:, self._i] - es
            absError = np.abs(error)
            mean = absError.mean()
            std = absError.std()
            originalErrorList.extend(error.reshape(-1).tolist())
            absErrorMeans.append(mean)
            absErrorStds.append(std)
        return absErrorMeans, absErrorStds, originalErrorList

    def crossValidate(self, N=8, variogram=None, **kwargs):
        '''
        Perform leave-one-out cross validation on sample points.

        :param N: integer, neighborhood size.
        :param variogram: variogram function.
        :return: tuple, tuple containing absolute mean error, absolute standard deviation error and origin error(ndarray).
        '''
        pass

    def __calcInnerVecs__(self):
        '''
        Compute vectors between sample points.

        :return: None.
        '''
        innerVecs = calcVecs(self.samples[:, :self._i], includeSelf=True)
        self.innerVecs = innerVecs.reshape((self.samples.shape[0], self.samples.shape[0], self._i))

    def __calcInnerVars__(self, variogram, isNest):
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogram is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
        elif isNest:
            self.innerVars = variogram(self.innerVecs)
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogram(self.innerVars)
        if type(self.innerVars)!=np.ndarray:# self.innerVars is a Tensor
            self.innerVars = self.innerVars.numpy()
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

    def __multiWorker__(self, points, N=8, variogram=None, workerNum=1, **kwargs):
        pfunc = partial(self.execute, N=N, variogram=variogram, workerNum=1, **kwargs)
        size = int(np.ceil(len(points) // workerNum)) + 1
        with Pool(workerNum) as p:
            result = p.map(pfunc, [points[i * size:(i + 1) * size] for i in range(workerNum)])
        properties = result[0][0]
        sigmas = result[0][1]
        result.pop(0)
        while len(result) > 0:
            pro, sig = result.pop(0)
            properties = np.append(properties, pro)
            sigmas = np.append(sigmas, sig)
        return properties, sigmas