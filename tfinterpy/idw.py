from tfinterpy.utils import kSplit
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
from functools import partial

class IDW:
    '''
    Inverse Distance Weighted interpolator.
    '''

    def __init__(self, samples, mode='2d'):
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
        if mode == '3d' or mode == '3D':
            self._i = 3

    def execute(self, points, N=8, alpha=2, workerNum=1):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param alpha: number, distance power factor.
        :param workerNum: By default, one process is used, and multi-process computation is used when wokerNum>1.
        :return: ndarray, one-dimensional array containing interpolation result.
        '''
        if workerNum>1:
            pfunc=partial(self.execute,N=N,alpha=alpha,workerNum=1)
            size=int(np.ceil(len(points)//workerNum))+1
            with Pool(workerNum) as p:
                result=p.map(pfunc,[points[i*size:(i+1)*size] for i in range(workerNum)])
            properties=result[0]
            result.pop(0)
            while len(result)>0:
                pro=result.pop(0)
                properties=np.append(properties,pro)
            return properties
        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=N, eps=0.0)
        properties = np.zeros((len(points)),dtype=np.float32)
        for idx, indice in enumerate(nbIdx):
            hs = nbd[idx]
            hs = hs ** alpha
            inv = 1 / (hs + 1e-8)
            total = np.sum(inv)
            weights = inv / total
            pro = np.dot(self.samples[:, self._i][indice], weights)
            properties[idx] = pro
        return properties

    def crossValidateKFold(self, K=10, N=8, alpha=2):
        '''
        Perform k-fold cross validation on sample points.

        :param K: integer.
        :param N: integer, neighborhood size.
        :param alpha: number, distance power factor.
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
            exe = IDW(p1, self.mode)
            es = exe.execute(p2[:, :self._i], N, alpha)
            error = p2[:, self._i] - es
            absError = np.abs(error)
            mean = absError.mean()
            std = absError.std()
            originalErrorList.append(error)
            absErrorMeans.append(mean)
            absErrorStds.append(std)
        return absErrorMeans, absErrorStds, originalErrorList

    def crossValidate(self, N=8, alpha=2):
        '''
        Perform leave-one-out cross validation on sample points.

        :param N: integer, neighborhood size.
        :param alpha: number, distance power factor.
        :return: tuple, tuple containing absolute mean error, absolute standard deviation error and origin error(ndarray).
        '''
        tree = cKDTree(self.samples[:, :self._i])
        nbd, nb_idx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        properties = np.zeros((len(self.samples)),dtype=np.float32)
        for idx, indice in enumerate(nb_idx):
            indice = indice[1:]
            hs = nbd[idx][1:]
            hs = hs ** alpha
            inv = 1 / (hs + 1e-8)
            total = np.sum(inv)
            weights = inv / total
            pro = np.dot(self.samples[:, self._i][indice], weights)
            properties[idx] = pro
        error = properties - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error
