import tensorflow as tf
import keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.utils import kSplit, calcVecs
from tfinterpy.tf.variogramLayer import NestVariogramLayer


def SKModel(n=8, variogramLayer=None, vecDim=2):
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
    def __init__(self, samples, mode='2d'):
        self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogramLayer=None, batch_size=1000):
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

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=N, eps=0.0)
        kmatList = []
        mvecList = []
        neighProList = []
        for idx, indice in enumerate(nbIdx):
            kmat = self.__getKrigeMat__(indice)
            if isNest:
                mvec = self.samples[indice, :self._i] - points[idx]
            else:
                mvec = nbd[idx]
            kmatList.append(kmat)
            mvecList.append(mvec)
            neighProList.append(self.samples[indice, self._i])
        kmatArr = np.array(kmatList)
        mvecArr = np.array(mvecList)
        neighProArr = np.array(neighProList)
        pros, sigmas = self.model.predict([kmatArr, mvecArr, neighProArr], batch_size=batch_size)
        return pros, sigmas

    def crossValidateKFold(self, K=10, N=8, variogramLayer=None):
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

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        kmatList = []
        mvecList = []
        neighProList = []
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            kmat = self.__getKrigeMat__(indice)
            if isNest:
                mvec = self.samples[indice, :self._i] - self.samples[idx, :self._i]
            else:
                mvec = nbd[idx][1:]
            kmatList.append(kmat)
            mvecList.append(mvec)
            neighProList.append(self.samples[indice, self._i])
        kmatArr = np.array(kmatList)
        mvecArr = np.array(mvecList)
        neighProArr = np.array(neighProList)
        pros, _ = self.model.predict([kmatArr, mvecArr, neighProArr], batch_size=1000)
        pros = pros.reshape(-1)
        error = pros - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error

    def __getKrigeMat__(self, indice):
        kmat = np.zeros((self.N, self.N))
        for i, idx1 in enumerate(indice):
            for j, idx2 in enumerate(indice):
                kmat[i, j] = self.innerVars[idx1, idx2]
        return kmat

    def __calcInnerVecs__(self):
        innerVecs = calcVecs(self.samples[:, :self._i], includeSelf=True)
        self.innerVecs = innerVecs.reshape((self.samples.shape[0], self.samples.shape[0], self._i))


def OKModel(n=8, variogramLayer=None, vecDim=2):
    mat1 = np.ones((n + 1, n + 1), dtype='float32')
    mat1[n] = 0
    mat1[:, n] = 0
    mat1 = tf.constant(mat1)
    mat2 = np.zeros((n + 1, n + 1), dtype='float32')
    mat2[n] = 1
    mat2[:, n] = 1
    mat2[n, n] = 0
    mat2 = tf.constant(mat2)

    mat3 = np.ones((n + 1, 1), dtype='float32')
    mat3[n] = 0
    mat3 = tf.constant(mat3)
    mat4 = np.zeros((n + 1, 1), dtype='float32')
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
    def __init__(self, samples, mode='2d'):
        self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogramLayer=None, batch_size=1000):
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

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=self.N, eps=0.0)
        kmatList = []
        mvecList = []
        neighProList = []
        if self.mode.lower() == '2d':
            addon = np.array([[0.0, 0.0]])
        else:
            addon = np.array([[0.0, 0.0, 0.0]])
        for idx, indice in enumerate(nbIdx):
            kmat = self.__getKrigeMat__(indice)
            if isNest:
                mvec = np.append(self.samples[indice, :self._i] - points[idx], addon, axis=0)
            else:
                mvec = np.append(nbd[idx], 1.0)
            kmatList.append(kmat)
            mvecList.append(mvec)
            neighProList.append(self.samples[indice, self._i])
        kmatArr = np.array(kmatList)
        mvecArr = np.array(mvecList)
        neighProArr = np.array(neighProList)
        pros, sigmas = self.model.predict([kmatArr, mvecArr, neighProArr], batch_size=batch_size)
        return pros, sigmas

    def crossValidateKFold(self, K=10, N=8, variogramLayer=None):
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

        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        kmatList = []
        mvecList = []
        neighProList = []
        if self.mode.lower() == '2d':
            addon = np.array([[0.0, 0.0]])
        else:
            addon = np.array([[0.0, 0.0, 0.0]])
        for idx, indice in enumerate(nbIdx):
            indice = indice[1:]
            kmat = self.__getKrigeMat__(indice)
            if isNest:
                mvec = np.append(self.samples[indice, :self._i] - self.samples[idx, :self._i], addon, axis=0)
            else:
                mvec = np.append(nbd[idx][1:], 1.0)
            kmatList.append(kmat)
            mvecList.append(mvec)
            neighProList.append(self.samples[indice, self._i])
        kmatArr = np.array(kmatList)
        mvecArr = np.array(mvecList)
        neighProArr = np.array(neighProList)
        pros, _ = self.model.predict([kmatArr, mvecArr, neighProArr], batch_size=1000)
        pros = pros.reshape(-1)
        error = pros - self.samples[:, self._i]
        absError = np.abs(error)
        mean = absError.mean()
        std = absError.std()
        return mean, std, error

    def __getKrigeMat__(self, indice):
        kmat = np.ones((self.N + 1, self.N + 1))
        kmat[self.N, self.N] = 0
        for i, idx1 in enumerate(indice):
            for j, idx2 in enumerate(indice):
                kmat[i, j] = self.innerVars[idx1, idx2]
        return kmat

    def __calcInnerVecs__(self):
        innerVecs = calcVecs(self.samples[:, :self._i], includeSelf=True)
        self.innerVecs = innerVecs.reshape((self.samples.shape[0], self.samples.shape[0], self._i))
