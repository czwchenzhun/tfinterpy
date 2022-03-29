import keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from scipy.spatial import cKDTree
import numpy as np
from tfinterpy.utils import kSplit


def IDWModel(n=8, alpha=2):
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
    def __init__(self, samples, mode='2d'):
        self.samples = samples
        self.mode = mode
        self._i = 2
        if mode == '3d' or mode == '3D':
            self._i = 3

    def execute(self, points, N=8, alpha=2, batch_size=1000):
        self.model = IDWModel(N, alpha)
        tree = cKDTree(self.samples[:, :self._i])
        nbd, nbIdx = tree.query(points, k=N, eps=0.0)
        hList = []
        neighProList = []
        for idx, indice in enumerate(nbIdx):
            hList.append(nbd[idx])
            neighProList.append(self.samples[indice, self._i])
        hArr = np.array(hList)
        neighProArr = np.array(neighProList)
        pros = self.model.predict([hArr, neighProArr], batch_size=batch_size)
        return pros

    def crossValidateKFold(self, K=10, N=8, alpha=2):
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
