import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.utils import kSplit


def IDWModel(n=8, alpha=2):
    '''
    Construction a keras model for Inverse Distance Weighted algorithm.

    :param n: integer, neighborhood size.
    :param alpha: number, distance power factor.
    :return: keras' Model object.
    '''
    h_ = layers.Input(shape=(n))
    pro = layers.Input(shape=(n))
    h = h_ ** alpha
    hinv = 1 / (h + 1e-8)
    total = K.sum(hinv, axis=1)
    total = layers.Reshape((1,))(total)
    weights = hinv / total
    estimate = layers.Dot(1)([pro, weights])
    model = Model(inputs=[h_, pro], outputs=estimate)
    return model


class TFIDW:
    '''
    Tensorflow version of Inverse Distance Weighted interpolator.
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

    def execute(self, points, N=8, alpha=2, batch_size=1000):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param alpha: number, distance power factor.
        :param batch_size: integer, size of each batch of data to be calculated.
        :return: ndarray, one-dimensional array containing interpolation result.
        '''
        self.model = IDWModel(N, alpha)
        tree = cKDTree(self.samples[:, :self._i])
        step = batch_size * 2
        num = int(np.ceil(len(points) / step))
        pros = np.empty((0, 1))
        for i in range(num):
            begin = i * step
            end = (i + 1) * step
            points_ = points[begin:end]
            nbd, nbIdx = tree.query(points_, k=N, eps=0.0)
            hList = []
            neighProList = []
            for idx, indice in enumerate(nbIdx):
                hList.append(nbd[idx])
                neighProList.append(self.samples[indice, self._i])
            hArr = np.array(hList)
            neighProArr = np.array(neighProList)
            pro = self.model.predict([hArr, neighProArr], batch_size=batch_size)
            pros = np.append(pros, pro)
        return pros

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
            exe = TFIDW(p1, self.mode)
            es = exe.execute(p2[:, :self._i], N, alpha)
            error = p2[:, self._i] - es
            absError = np.abs(error)
            mean = absError.mean()
            std = absError.std()
            originalErrorList.extend(error.reshape(-1).tolist())
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
        self.model = IDWModel(N, alpha)
        tree = cKDTree(self.samples[:, :self._i])
        nbd, nb_idx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        hList = []
        neighProList = []
        for idx, indice in enumerate(nb_idx):
            hList.append(nbd[idx][1:])
            neighProList.append(self.samples[indice[1:], self._i])
        hArr = np.array(hList)
        neighProArr = np.array(neighProList)
        pros = self.model.predict([hArr, neighProArr], batch_size=1000)
        pros = pros.reshape(-1)
        error = pros - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error
