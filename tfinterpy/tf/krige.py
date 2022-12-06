import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.utils import kSplit, calcVecs
from tfinterpy.tf.variogramLayer import NestVariogramLayer
from tfinterpy.settings import dtype
from multiprocessing import Pool
from functools import partial
import warnings

tf.keras.backend.set_floatx(dtype)

class KMatLayer(layers.Layer):
    def __init__(self, innerVars):
        super(KMatLayer,self).__init__()
        self.innerVars = innerVars

    @tf.function(jit_compile=True)
    def call(self,indices):
        return tf.vectorized_map(lambda indice: tf.gather(tf.gather(self.innerVars, indice), indice, axis=1), indices)

class MVecLayer(layers.Layer):
    def __init__(self, sampleLocs):
        super(MVecLayer,self).__init__()
        self.sampleLocs = sampleLocs

    @tf.function(jit_compile=True)
    def call(self,indices, points):
        return tf.vectorized_map(lambda i: tf.gather(self.sampleLocs, indices[i]) - points[i], tf.range(tf.shape(indices)[0]))

class IndiceLayer(layers.Layer):
    def __init__(self, data):
        super(IndiceLayer, self).__init__()
        self.data = data

    @tf.function(jit_compile=True)
    def call(self, indices):
        return tf.vectorized_map(lambda indice: tf.gather(self.data, indice), indices)

class BatchConcatenate(layers.Layer):
    def __init__(self):
        super(BatchConcatenate, self).__init__()

    @tf.function(jit_compile=True)
    def call(self, x, y, axis=0):
        return tf.vectorized_map(lambda a:tf.concat([a,y], axis), x)

def SKModel(innerVars, sampleLocs, samplePros, n=8, variogramLayer=None):
    '''
    Construction a keras model for Simple Kriging algorithm.

    :param innerVars: ndarray, the square of the difference between the sampling point properties.
    :param sampleLocs: ndarray, sampling point coordinates.
    :param samplePros: ndarray, sampling point properties.
    :param n: integer, neighborhood size.
    :param variogramLayer: keras' layer, layer representing a variogram function.
    :return: keras' Model object.
    '''
    innerVars = tf.convert_to_tensor(innerVars)
    sampleLocs = tf.convert_to_tensor(sampleLocs)
    samplePros = tf.convert_to_tensor(samplePros)
    indices=layers.Input(shape=(n,), dtype=tf.int32)
    points=layers.Input(shape=(sampleLocs.shape[1],))
    kmat=KMatLayer(innerVars)(indices)
    mvec=MVecLayer(sampleLocs)(indices, points)
    pro=IndiceLayer(samplePros)(indices)
    if variogramLayer != None:
        mvec = variogramLayer(mvec)
    else:
        mvec = tf.linalg.norm(mvec, axis=2)
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
    model = Model(inputs=[indices, points], outputs=[estimate, sigma])
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
        if samples.dtype!=dtype:
            self.samples = samples.astype(dtype)
        else:
            self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogramLayer=None, batch_size=10000, workerNum=1, device='/CPU:0'):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param variogramLayer: keras' layer, layer representing a variogram function.
        :param batch_size: integer, size of each batch of data to be calculated.
        :param workerNum: By default, one process is used, and multi-process computation is used when wokerNum>1.
            Notice! If GPU device is specified, the multi-process cannot be enabled.
        :param device: Specified computing device, default value is '/CPU:0'.
        :return: tuple, tuple containing tow ndarray.
            The first ndarray representing the interpolation result,
            the second ndarray representing the kriging variance.
        '''
        if workerNum>1:
            warnings.warn("Multi-process tasks are not recommended when the data volume is small")
            if 'GPU' in device.upper():
                warnings.warn("It is not recommended to use the GPU for multi-process tasks, and the speed may actually decrease!")
            # assert 'GPU' not in device.upper(), "multi-process with GPU is not supported!"
            pfunc=partial(self.execute,N=N,variogramLayer=variogramLayer,batch_size=batch_size,workerNum=1, device=device)
            size=int(np.ceil(len(points)//workerNum))+1
            with Pool(workerNum) as p:
                result=p.map(pfunc,[points[i*size:(i+1)*size] for i in range(workerNum)])
            properties=result[0][0]
            sigmas=result[0][1]
            result.pop(0)
            while len(result)>0:
                pro,sig=result.pop(0)
                properties=np.append(properties,pro)
                sigmas=np.append(sigmas,sig)
            return properties,sigmas
        with tf.device(device):
            self.N = N
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

            self.model = SKModel(self.innerVars, self.samples[:,:self._i], self.samples[:, self._i], N, variogramLayer)
            tree = cKDTree(self.samples[:, :self._i])
            step = batch_size * 3
            num = int(np.ceil(len(points) / step))
            pros = np.empty((0, 1),dtype=np.float32)
            sigmas = np.empty((0, 1),dtype=np.float32)
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
        self.model = SKModel(self.innerVars, self.samples[:, :self._i], self.samples[:, self._i], N, variogramLayer)
        tree = cKDTree(self.samples[:, :self._i])
        _, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        pros, _ = self.model.predict([nbIdx[:, 1:], self.samples[:, :self._i]], batch_size=10000)
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

def OKModel(innerVars, sampleLocs, samplePros, n=8, variogramLayer=None):
    '''
    Construction a keras model for Ordinary Kriging algorithm.

    :param innerVars: ndarray, the square of the difference between the sampling point properties.
    :param sampleLocs: ndarray, sampling point coordinates.
    :param samplePros: ndarray, sampling point properties.
    :param n: integer, neighborhood size.
    :param variogramLayer: keras' layer, layer representing a variogram function.
    :return: keras' Model object.
    '''
    mat1 = np.ones((n + 1, n + 1),dtype=dtype)
    mat1[n, n] = 0
    mat1 = tf.constant(mat1, dtype=dtype)

    tmp = np.ones((innerVars.shape[0] + 1, innerVars.shape[0] + 1), dtype=dtype)
    tmp[:innerVars.shape[0], :innerVars.shape[0]] = innerVars
    innerVars = tmp

    innerVars = tf.convert_to_tensor(innerVars)
    sampleLocs = tf.convert_to_tensor(sampleLocs)
    samplePros = tf.convert_to_tensor(samplePros)

    indices = layers.Input(shape=(n,), dtype="int32")
    points = layers.Input(shape=(sampleLocs.shape[1],))

    mvec = MVecLayer(sampleLocs)(indices, points)
    pro = IndiceLayer(samplePros)(indices)

    tmp2 = tf.zeros((1,),dtype="int32")
    tmp2 = tmp2 + (innerVars.shape[0]-1)

    indices_ = BatchConcatenate()(indices, tmp2)
    kmat = KMatLayer(innerVars)(indices_)
    kmat = kmat * mat1

    if variogramLayer != None:
        mvec = variogramLayer(mvec)
        mvec = BatchConcatenate()(mvec, tf.ones((1, 1), dtype=dtype))
    else:
        mvec = tf.linalg.norm(mvec, axis=2)
        mvec = BatchConcatenate()(mvec, tf.ones((1,), dtype=dtype))
        mvec = layers.Reshape((n + 1, 1))(mvec)
    kmatInv = tf.linalg.pinv(kmat)
    lambvec = layers.Dot(1)([kmatInv, mvec])
    estimate = layers.Dot(1)([pro, lambvec[:, :n]])
    lambvec = layers.Reshape((n + 1,))(lambvec)
    mvec = layers.Reshape((n + 1,))(mvec)
    sigma = layers.Dot(1)([lambvec, mvec])
    model = Model(inputs=[indices, points], outputs=[estimate, sigma])
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
        if samples.dtype!=dtype:
            self.samples = samples.astype(dtype)
        else:
            self.samples = samples
        self.mode = mode
        self._i = 2
        if mode.lower() == '3d':
            self._i = 3
        self.innerVecs = None

    def execute(self, points, N=8, variogramLayer=None, batch_size=10000, workerNum=1, device='/CPU:0'):
        '''
        Perform interpolation for points and return result values.

        :param points: ndarray, array containing all the coordinate points to be interpolated.
        :param N: integer, neighborhood size.
        :param variogramLayer: keras' layer, layer representing a variogram function.
        :param batch_size: integer, size of each batch of data to be calculated.
        :param workerNum: By default, one process is used, and multi-process computation is used when wokerNum>1.
            Notice! If GPU device is specified, the multi-process cannot be enabled.
        :param device: Specified computing device, default value is '/CPU:0'.
        :return: tuple, tuple containing tow ndarray.
            The first ndarray representing the interpolation result,
            the second ndarray representing the kriging variance.
        '''
        if workerNum>1:
            warnings.warn("Multi-process tasks are not recommended when the data volume is small")
            if 'GPU' in device.upper():
                warnings.warn("It is not recommended to use the GPU for multi-process tasks, and the speed may actually decrease!")
            # assert 'GPU' not in device.upper(), "multi-process with GPU is not supported!"
            pfunc=partial(self.execute,N=N,variogramLayer=variogramLayer,batch_size=batch_size,workerNum=1, device=device)
            size=int(np.ceil(len(points)//workerNum))+1
            with Pool(workerNum) as p:
                result=p.map(pfunc,[points[i*size:(i+1)*size] for i in range(workerNum)])
            properties=result[0][0]
            sigmas=result[0][1]
            result.pop(0)
            while len(result)>0:
                pro,sig=result.pop(0)
                properties=np.append(properties,pro)
                sigmas=np.append(sigmas,sig)
            return properties,sigmas
        with tf.device(device):
            self.N = N
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

            self.model = OKModel(self.innerVars, self.samples[:,:self._i], self.samples[:,self._i], N, variogramLayer)
            tree = cKDTree(self.samples[:, :self._i])
            step = batch_size * 3
            num = int(np.ceil(len(points) / step))
            pros = np.empty((0, 1),dtype=np.float32)
            sigmas = np.empty((0, 1),dtype=np.float32)
            for i in range(num):
                begin = i * step
                end = (i + 1) * step
                if end > len(points):
                    end = len(points)
                points_ = points[begin:end]
                _, nbIdx = tree.query(points_, k=self.N, eps=0.0)
                pro, sigma = self.model.predict([nbIdx, points_], batch_size=batch_size)
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
        self.model = OKModel(self.innerVars, self.samples[:, :self._i], self.samples[:, self._i], N, variogramLayer)
        tree = cKDTree(self.samples[:, :self._i])
        _, nbIdx = tree.query(self.samples[:, :self._i], k=N + 1, eps=0.0)
        pros, _ = self.model.predict([nbIdx[:, 1:], self.samples[:, :self._i]], batch_size=10000)
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
