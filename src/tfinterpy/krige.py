from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.utils import kSplit, calcVecs
from tfinterpy.variogram import NestVariogram


class SK:
    '''
    Simple Kriging interpolator.
    '''

    def __init__(self, samples, mode="2d"):
        '''
        Initialize the interpolator using sample points.

        :param samples: ndarray, array containing all sample points. The last column must be the properties.
            For the case of two-dimensional interpolation, where each item is represented by [x,y,property].
            For the case of three-dimensional interpolation, where each item is represented by [x,y,z,property].
        :param mode: str, '2d' or '3d'.
        '''
        self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogram=None):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param variogram: variogram function or nest variogram object, default None.
            A linear variogram (lambda x:x) is used when the variogram is None.
        :return: tuple, tuple containing tow ndarray.
            The first ndarray representing the interpolation result,
            the second ndarray representing the kriging variance.
        '''
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogram is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            variogram = lambda x: x
        elif variogram.__class__ == NestVariogram:
            self.innerVars = variogram(self.innerVecs)
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogram(self.innerVars)
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=N, eps=0.0)
        properties = np.zeros((len(points)))
        sigmas = np.zeros((len(points)))
        for idx, indice in enumerate(nbIdx):
            kmat = self.innerVars[indice][:, indice]
            if variogram.__class__ == NestVariogram:
                mvec = self.samples[indice, :self._i] - points[idx]
                mvec = variogram(mvec)
            else:
                mvec = variogram(nbd[idx])
            try:
                lambvec = np.linalg.inv(kmat).dot(mvec)
            except:
                lambvec = np.linalg.pinv(kmat).dot(mvec)
            pro = np.dot(self.samples[:, self._i][indice], lambvec)
            properties[idx] = pro + np.mean(self.samples[:, self._i][indice]) * (1 - np.sum(lambvec))
            sigmas[idx] = np.dot(lambvec, mvec)
        return properties, sigmas

    def crossValidateKFold(self, K=10, N=8, variogram=None):
        '''
        Perform k-fold cross validation on sample points.

        :param K: integer.
        :param N: integer, neighborhood size.
        :param variogram: variogram function or nest variogram object, default None.
            A linear variogram (lambda x:x) is used when the variogram is None.
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
            exe = SK(p1, self.mode)
            es, _ = exe.execute(p2[:, :self._i], N, variogram)
            error = p2[:, self._i] - es
            absError = np.abs(error)
            mean = absError.mean()
            std = absError.std()
            originalErrorList.append(error)
            absErrorMeans.append(mean)
            absErrorStds.append(std)
        return absErrorMeans, absErrorStds, originalErrorList

    def crossValidate(self, N=8, variogram=None):
        '''
        Perform leave-one-out cross validation on sample points.

        :param N: integer, neighborhood size.
        :param variogram: variogram function or nest variogram object, default None.
            A linear variogram (lambda x:x) is used when the variogram is None.
        :return: tuple, tuple containing absolute mean error, absolute standard deviation error and origin error(ndarray).
        '''
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogram is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            variogram = lambda x: x
        elif variogram.__class__ == NestVariogram:
            self.innerVars = variogram(self.innerVecs)
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogram(self.innerVars)
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        properties = np.zeros((len(self.samples)))
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            kmat = self.innerVars[indice][:, indice]
            if variogram.__class__ == NestVariogram:
                mvec = self.samples[indice, :self._i] - self.samples[idx, :self._i]
                mvec = variogram(mvec)
            else:
                mvec = variogram(nbd[idx][1:])
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

    def __calcInnerVecs__(self):
        '''
        Compute vectors between sample points.

        :return: None.
        '''
        innerVecs = calcVecs(self.samples[:, :self._i], includeSelf=True)
        self.innerVecs = innerVecs.reshape((self.samples.shape[0], self.samples.shape[0], self._i))


class OK:
    '''
    Ordinary Kriging interpolator.
    '''

    def __init__(self, samples, mode="2d"):
        '''
        Initialize the interpolator using sample points.

        :param samples: ndarray, array containing all sample points. The last column must be the properties.
            For the case of two-dimensional interpolation, where each item is represented by [x,y,property].
            For the case of three-dimensional interpolation, where each item is represented by [x,y,z,property].
        :param mode: str, '2d' or '3d'.
        '''
        self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogram=None):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param variogram: variogram function or nest variogram object, default None.
            A linear variogram (lambda x:x) is used when the variogram is None.
        :return: tuple, tuple containing tow ndarray.
            The first ndarray representing the interpolation result,
            the second ndarray representing the kriging variance.
        '''
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogram is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            variogram = lambda x: x
        elif variogram.__class__ == NestVariogram:
            self.innerVars = variogram(self.innerVecs)
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogram(self.innerVars)
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=N, eps=0.0)
        properties = np.zeros((len(points)))
        sigmas = np.zeros((len(points)))
        kmat = np.ones((N + 1, N + 1))
        kmat[N, N] = 0.0
        mvec = np.ones((N + 1,))
        for idx, indice in enumerate(nbIdx):
            kmat[:N, :N] = self.innerVars[indice][:, indice]
            if variogram.__class__ == NestVariogram:
                vecs = self.samples[indice, :self._i] - points[idx]
                mvec[:N] = variogram(vecs)
            else:
                mvec[:N] = variogram(nbd[idx])
            # np.fill_diagonal(kmat,0.0)
            try:
                lambvec = np.linalg.inv(kmat).dot(mvec)
            except:
                lambvec = np.linalg.pinv(kmat).dot(mvec)
            properties[idx] = np.dot(self.samples[indice, self._i], lambvec[:N])
            sigmas[idx] = np.dot(lambvec, mvec)
        return properties, sigmas

    def crossValidateKFold(self, K=10, N=8, variogram=None):
        '''
        Perform k-fold cross validation on sample points.

        :param K: integer.
        :param N: integer, neighborhood size.
        :param variogram: variogram function or nest variogram object, default None.
            A linear variogram (lambda x:x) is used when the variogram is None.
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
            exe = OK(p1, self.mode)
            es, _ = exe.execute(p2[:, :self._i], N, variogram)
            error = p2[:, self._i] - es
            absError = np.abs(error)
            mean = absError.mean()
            std = absError.std()
            originalErrorList.append(error)
            absErrorMeans.append(mean)
            absErrorStds.append(std)
        return absErrorMeans, absErrorStds, originalErrorList

    def crossValidate(self, N=8, variogram=None):
        '''
        Perform leave-one-out cross validation on sample points.

        :param N: integer, neighborhood size.
        :param variogram: variogram function or nest variogram object, default None.
            A linear variogram (lambda x:x) is used when the variogram is None.
        :return: tuple, tuple containing absolute mean error, absolute standard deviation error and origin error(ndarray).
        '''
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogram is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            variogram = lambda x: x
        elif variogram.__class__ == NestVariogram:
            self.innerVars = variogram(self.innerVecs)
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogram(self.innerVars)
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        properties = np.zeros((len(self.samples)))
        kmat = np.ones((N + 1, N + 1))
        kmat[N, N] = 0.0
        mvec = np.ones((N + 1,))
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            kmat[:N, :N] = self.innerVars[indice][:, indice]
            if variogram.__class__ == NestVariogram:
                vecs = self.samples[indice, :self._i] - self.samples[idx, :self._i]
                mvec[:N] = variogram(vecs)
            else:
                mvec[:N] = variogram(nbd[idx][1:])
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

    def __calcInnerVecs__(self):
        '''
        Compute vectors between sample points.

        :return: None.
        '''
        innerVecs = calcVecs(self.samples[:, :self._i], includeSelf=True)
        self.innerVecs = innerVecs.reshape((self.samples.shape[0], self.samples.shape[0], self._i))
