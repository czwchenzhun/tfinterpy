from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.utils import kSplit, calcVecs
from tfinterpy.variogram import NestVariogram


class SK:
    def __init__(self, samples, mode="2d"):
        self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogram=None):
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

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=N, eps=0.0)
        properties = np.zeros((len(points)))
        sigmas = np.zeros((len(points)))
        self.kmat = np.zeros((N, N))
        for idx, indice in enumerate(nbIdx):
            self.__getKrigeMat__(indice)
            if variogram.__class__ == NestVariogram:
                mvec = self.samples[indice, :self._i] - points[idx]
                mvec = variogram(mvec)
            else:
                mvec = variogram(nbd[idx])
            try:
                lambvec = np.linalg.inv(self.kmat).dot(mvec)
            except:
                lambvec = np.linalg.pinv(self.kmat).dot(mvec)
            pro = np.dot(self.samples[:, self._i][indice], lambvec)
            properties[idx] = pro + np.mean(self.samples[:, self._i][indice]) * (1 - np.sum(lambvec))
            sigmas[idx] = np.dot(lambvec, mvec)
        return properties, sigmas

    def crossValidateKFold(self, K=10, N=8, variogram=None):
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

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        self.kmat = np.zeros((N, N))
        self.mvec = np.zeros((N))
        properties = np.zeros((len(self.samples)))
        self.kmat = np.zeros((N, N))
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            self.__getKrigeMat__(indice)
            if variogram.__class__ == NestVariogram:
                mvec = self.samples[indice, :self._i] - self.samples[idx, :self._i]
                mvec = variogram(mvec)
            else:
                mvec = variogram(nbd[idx][1:])
            try:
                lambvec = np.linalg.inv(self.kmat).dot(mvec)
            except:
                lambvec = np.linalg.pinv(self.kmat).dot(mvec)
            pro = np.dot(self.samples[:, self._i][indice], lambvec)
            properties[idx] = pro + np.mean(self.samples[:, self._i][indice]) * (1 - np.sum(lambvec))
        error = properties - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error

    def __getKrigeMat__(self, indice):
        for i, idx1 in enumerate(indice):
            for j, idx2 in enumerate(indice):
                self.kmat[i, j] = self.innerVars[idx1, idx2]

    def __calcInnerVecs__(self):
        innerVecs = calcVecs(self.samples[:, :self._i], includeSelf=True)
        self.innerVecs = innerVecs.reshape((self.samples.shape[0], self.samples.shape[0], self._i))


class OK:
    def __init__(self, samples, mode="2d"):
        self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogram=None):
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

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=N, eps=0.0)
        properties = np.zeros((len(points)))
        sigmas = np.zeros((len(points)))
        self.kmat = np.ones((N + 1, N + 1))
        self.kmat[N, N] = 0.0
        self.mvec = np.ones((N + 1,))
        for idx, indice in enumerate(nbIdx):
            self.__getKrigeMat__(indice)
            if variogram.__class__ == NestVariogram:
                mvec = self.samples[indice, :self._i] - points[idx]
                self.mvec[:N] = variogram(mvec)
            else:
                self.mvec[:N] = variogram(nbd[idx])
            # np.fill_diagonal(self.kmat,0.0)
            try:
                lambvec = np.linalg.inv(self.kmat).dot(self.mvec)
            except:
                lambvec = np.linalg.pinv(self.kmat).dot(self.mvec)
            properties[idx] = np.dot(self.samples[indice, self._i], lambvec[:N])
            sigmas[idx] = np.dot(lambvec, self.mvec)
        return properties, sigmas

    def crossValidateKFold(self, K=10, N=8, variogram=None):
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

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        properties = np.zeros((len(self.samples)))
        self.kmat = np.ones((N + 1, N + 1))
        self.kmat[N, N] = 0.0
        self.mvec = np.ones((N + 1))
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            self.__getKrigeMat__(indice)
            if variogram.__class__ == NestVariogram:
                mvec = self.samples[indice, :self._i] - self.samples[idx, :self._i]
                self.mvec[:N] = variogram(mvec)
            else:
                self.mvec[:N] = variogram(nbd[idx][1:])
            # np.fill_diagonal(self.kmat,0.0)
            try:
                lambvec = np.linalg.inv(self.kmat).dot(self.mvec)
            except:
                lambvec = np.linalg.pinv(self.kmat).dot(self.mvec)
            properties[idx] = np.dot(self.samples[:, self._i][indice], lambvec[:N])
        error = properties - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error

    def __getKrigeMat__(self, indice):
        for i, idx1 in enumerate(indice):
            for j, idx2 in enumerate(indice):
                self.kmat[i, j] = self.innerVars[idx1, idx2]

    def __calcInnerVecs__(self):
        innerVecs = calcVecs(self.samples[:, :self._i], includeSelf=True)
        self.innerVecs = innerVecs.reshape((self.samples.shape[0], self.samples.shape[0], self._i))
