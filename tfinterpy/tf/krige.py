import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.utils import kSplit, calcVecs
from tfinterpy.tf.variogramLayer import NestVariogramLayer

# tf.keras.backend.set_floatx('float64')

def SKModel(n=8, variogramLayer=None, vecDim=2):
    '''
    Construction a keras model for Simple Kriging algorithm.

    :param n: integer, neighborhood size.
    :param variogramLayer: keras' layer, layer representing a variogram function.
    :param vecDim: integer, the dimension of the vector to be calculated.
    :return: keras' Model object.
    '''
    kmat = layers.Input(shape=(n, n))
    if variogramLayer != None and variogramLayer.__class__ == NestVariogramLayer:
        mvec_ = layers.Input(shape=(n, vecDim))
    else:
        mvec_ = layers.Input(shape=(n))
    pro = layers.Input(shape=(n))
    if variogramLayer != None:
        mvec = variogramLayer(mvec_)
    else:
        mvec = mvec_
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
    model = Model(inputs=[kmat, mvec_, pro], outputs=[estimate, sigma])
    return model


class TFSK:
    '''
    Tensorflow version of Simple Kriging interpolator.
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
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogramLayer=None, batch_size=10000):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param variogramLayer: keras' layer, layer representing a variogram function.
        :param batch_size: integer, size of each batch of data to be calculated.
        :return: tuple, tuple containing tow ndarray.
            The first ndarray representing the interpolation result,
            the second ndarray representing the kriging variance.
        '''
        self.N = N
        self.model = SKModel(N, variogramLayer, self._i)
        isNest = variogramLayer.__class__ == NestVariogramLayer
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogramLayer is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
        elif isNest:
            self.innerVars = variogramLayer(self.innerVecs).numpy()
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogramLayer(self.innerVars).numpy()
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

        tree = cKDTree(self.samples[:, :self._i])
        step = batch_size * 3
        num = int(np.ceil(len(points) / step))
        pros = np.empty((0, 1))
        sigmas = np.empty((0, 1))
        init = False
        for i in range(num):
            begin = i * step
            end = (i + 1) * step
            if end > len(points):
                end = len(points)
            if not init or i == num - 1:
                kmatArr = np.zeros((end - begin, N, N))
                if isNest:
                    mvecArr = np.zeros((end - begin, N, self._i))
                else:
                    mvecArr = np.zeros((end - begin, N))
                neighProArr = np.zeros((end - begin, N))
                init = True
            points_ = points[begin:end]
            nbd, nbIdx = tree.query(points_, k=N, eps=0.0)
            for idx, indice in enumerate(nbIdx):
                kmatArr[idx] = self.innerVars[indice][:, indice]
                if isNest:
                    mvecArr[idx] = self.samples[indice, :self._i] - points_[idx]
                else:
                    mvecArr[idx] = nbd[idx]
                neighProArr[idx] = self.samples[indice, self._i]
            pro, sigma = self.model.predict([kmatArr, mvecArr, neighProArr], batch_size=batch_size)
            pros = np.append(pros, pro)
            sigmas = np.append(sigmas, sigma)
        return pros, sigmas

    def crossValidateKFold(self, K=10, N=8, variogramLayer=None):
        '''
        Perform k-fold cross validation on sample points.

        :param K: integer.
        :param N: integer, neighborhood size.
        :param variogramLayer: keras' layer, layer representing a variogram function.
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
            exe = TFSK(p1, self.mode)
            es, _ = exe.execute(p2[:, :self._i], N, variogramLayer)
            error = p2[:, self._i] - es
            absError = np.abs(error)
            mean = absError.mean()
            std = absError.std()
            originalErrorList.extend(error.reshape(-1).tolist())
            absErrorMeans.append(mean)
            absErrorStds.append(std)
        return absErrorMeans, absErrorStds, originalErrorList

    def crossValidate(self, N=8, variogramLayer=None):
        '''
        Perform leave-one-out cross validation on sample points.

        :param N: integer, neighborhood size.
        :param variogramLayer: keras' layer, layer representing a variogram function.
        :return: tuple, tuple containing absolute mean error, absolute standard deviation error and origin error(ndarray).
        '''
        self.N = N
        self.model = SKModel(N, variogramLayer, self._i)
        isNest = variogramLayer.__class__ == NestVariogramLayer
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogramLayer is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
        elif isNest:
            self.innerVars = variogramLayer(self.innerVecs).numpy()
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogramLayer(self.innerVars).numpy()
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        L = len(self.samples)
        kmatArr = np.zeros((L, N, N))
        if isNest:
            mvecArr = np.zeros((L, N, self._i))
        else:
            mvecArr = np.zeros((L, N))
        neighProArr = np.zeros((L, N))
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            kmatArr[idx] = self.innerVars[indice][:, indice]
            if isNest:
                mvecArr[idx] = self.samples[indice, :self._i] - self.samples[idx, :self._i]
            else:
                mvecArr[idx] = nbd[idx][1:]
            neighProArr[idx] = self.samples[indice, self._i]
        pros, _ = self.model.predict([kmatArr, mvecArr, neighProArr], batch_size=10000)
        pros = pros.reshape(-1)
        error = pros - self.samples[:, self._i]
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


def OKModel(n=8, variogramLayer=None, vecDim=2):
    '''
    Construction a keras model for Ordinary Kriging algorithm.

    :param n: integer, neighborhood size.
    :param variogramLayer: keras' layer, layer representing a variogram function.
    :param vecDim: integer, the dimension of the vector to be calculated.
    :return: keras' Model object.
    '''
    mat1 = np.ones((n + 1, n + 1))
    mat1[n] = 0
    mat1[:, n] = 0
    mat1 = tf.constant(mat1)
    mat2 = np.zeros((n + 1, n + 1))
    mat2[n] = 1
    mat2[:, n] = 1
    mat2[n, n] = 0
    mat2 = tf.constant(mat2)

    mat3 = np.ones((n + 1, 1))
    mat3[n] = 0
    mat3 = tf.constant(mat3)
    mat4 = np.zeros((n + 1, 1))
    mat4[n] = 1
    mat4 = tf.constant(mat4)

    kmat_ = layers.Input(shape=(n + 1, n + 1))
    if variogramLayer != None and variogramLayer.__class__ == NestVariogramLayer:
        mvec_ = layers.Input(shape=(n + 1, vecDim))
    else:
        mvec_ = layers.Input(shape=(n + 1))
    pro = layers.Input(shape=(n,))
    if variogramLayer != None:
        mvec = variogramLayer(mvec_)
        mvec = layers.Reshape((n + 1, 1))(mvec)

        kmat = kmat_
        kmat = kmat * mat1
        kmat = kmat + mat2
        mvec = mvec * mat3
        mvec = mvec + mat4
    else:
        kmat = kmat_
        mvec = mvec_
        mvec = layers.Reshape((n + 1, 1))(mvec)
    kmatInv = tf.linalg.pinv(kmat)
    lambvec = layers.Dot(1)([kmatInv, mvec])
    estimate = layers.Dot(1)([pro, lambvec[:, :n]])

    lambvec = layers.Reshape((n + 1,))(lambvec)
    mvec = layers.Reshape((n + 1,))(mvec)
    sigma = layers.Dot(1)([lambvec, mvec])
    model = Model(inputs=[kmat_, mvec_, pro], outputs=[estimate, sigma])
    return model


class TFOK:
    '''
    Tensorflow version of Ordinary Kriging interpolator.
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
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogramLayer=None, batch_size=10000):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param variogramLayer: keras' layer, layer representing a variogram function.
        :param batch_size: integer, size of each batch of data to be calculated.
        :return: tuple, tuple containing tow ndarray.
            The first ndarray representing the interpolation result,
            the second ndarray representing the kriging variance.
        '''
        self.N = N
        self.model = OKModel(N, variogramLayer, self._i)
        isNest = variogramLayer.__class__ == NestVariogramLayer
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogramLayer is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
        elif isNest:
            self.innerVars = variogramLayer(self.innerVecs).numpy()
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogramLayer(self.innerVars).numpy()
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

        tree = cKDTree(self.samples[:, :self._i])
        step = batch_size * 3
        num = int(np.ceil(len(points) / step))
        pros = np.empty((0, 1))
        sigmas = np.empty((0, 1))
        init = False
        for i in range(num):
            begin = i * step
            end = (i + 1) * step
            if end > len(points):
                end = len(points)
            if not init or i == num - 1:
                kmatArr = np.ones((end - begin, N + 1, N + 1))
                for j in range(end - begin):
                    kmatArr[j, N, N] = 0.0
                if isNest:
                    mvecArr = np.zeros((end - begin, N + 1, self._i))
                else:
                    mvecArr = np.ones((end - begin, N + 1))
                neighProArr = np.zeros((end - begin, N))
                init = True
            points_ = points[begin:end]
            nbd, nbIdx = tree.query(points_, k=self.N, eps=0.0)
            for idx, indice in enumerate(nbIdx):
                kmatArr[idx, :N, :N] = self.innerVars[indice][:, indice]
                if isNest:
                    mvecArr[idx, :N] = self.samples[indice, :self._i] - points_[idx]
                else:
                    mvecArr[idx, :N] = nbd[idx]
                neighProArr[idx] = self.samples[indice, self._i]
            pro, sigma = self.model.predict([kmatArr, mvecArr, neighProArr], batch_size=batch_size)
            pros = np.append(pros, pro)
            sigmas = np.append(sigmas, sigma)
        return pros, sigmas

    def crossValidateKFold(self, K=10, N=8, variogramLayer=None):
        '''
        Perform k-fold cross validation on sample points.

        :param K: integer.
        :param N: integer, neighborhood size.
        :param variogramLayer: keras' layer, layer representing a variogram function.
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
            exe = TFOK(p1, self.mode)
            es, _ = exe.execute(p2[:, :self._i], N, variogramLayer)
            error = p2[:, self._i] - es
            absError = np.abs(error)
            mean = absError.mean()
            std = absError.std()
            originalErrorList.extend(error.reshape(-1).tolist())
            absErrorMeans.append(mean)
            absErrorStds.append(std)
        return absErrorMeans, absErrorStds, originalErrorList

    def crossValidate(self, N=8, variogramLayer=None):
        '''
        Perform leave-one-out cross validation on sample points.

        :param N: integer, neighborhood size.
        :param variogramLayer: keras' layer, layer representing a variogram function.
        :return: tuple, tuple containing absolute mean error, absolute standard deviation error and origin error(ndarray).
        '''
        self.N = N
        self.model = OKModel(N, variogramLayer, self._i)
        isNest = variogramLayer.__class__ == NestVariogramLayer
        if self.innerVecs is None:
            self.__calcInnerVecs__()
        if variogramLayer is None:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
        elif isNest:
            self.innerVars = variogramLayer(self.innerVecs).numpy()
        else:
            self.innerVars = np.linalg.norm(self.innerVecs, axis=2)
            self.innerVars = variogramLayer(self.innerVars).numpy()
        if self.innerVars.shape[-1] == 1:
            self.innerVars = self.innerVars.reshape(self.innerVars.shape[:-1])

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        L = len(self.samples)
        kmatArr = np.ones((L, N + 1, N + 1))
        for j in range(L):
            kmatArr[j, N, N] = 0.0
        if isNest:
            mvecArr = np.zeros((L, N + 1, self._i))
        else:
            mvecArr = np.ones((L, N + 1))
        neighProArr = np.zeros((L, N))
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            kmatArr[idx, :N, :N] = self.innerVars[indice][:, indice]
            if isNest:
                mvecArr[idx, :N] = self.samples[indice, :self._i] - self.samples[idx, :self._i]
            else:
                mvecArr[idx, :N] = nbd[idx][1:]
            neighProArr[idx] = self.samples[indice, self._i]
        pros, _ = self.model.predict([kmatArr, mvecArr, neighProArr], batch_size=10000)
        pros = pros.reshape(-1)
        error = pros - self.samples[:, self._i]
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
