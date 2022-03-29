import numpy as np


def h2(p1, p2):
    diff = np.square(np.subtract(p1, p2))
    return np.sum(diff, axis=-1)


def h(p1, p2):
    return np.sqrt(h2(p1, p2))


def ha(p1, p2):
    diff = np.subtract(p1, p2)
    h = np.sqrt(np.sum(np.square(diff), axis=-1))
    if diff.ndim == 1:
        angle = np.arctan2(diff[1], diff[0])
        angle = angle + np.pi if angle < 0 else angle
    else:
        angle = np.arctan2(diff[:, 1], diff[:, 0])
        angle[angle < 0] += np.pi
    return h, angle


def calcHV(data, dim=2, attrib_col=2):
    if dim == 3 and attrib_col < 3:
        attrib_col = 3
    arr = np.empty((0, 2), dtype="float32")
    size = data.shape[0]
    for i in range(size - 1):
        dis = h(data[i + 1:, :dim], data[i, :dim])
        halfvar = 0.5 * ((data[i + 1:, attrib_col] - data[i, attrib_col]) ** 2)
        dis.resize((dis.shape[0], 1))
        halfvar.resize((halfvar.shape[0], 1))
        arr = np.append(arr, np.concatenate([dis, halfvar], axis=1), axis=0)
    return arr


def calcHAV(data, attrib_col=2):
    dim = 2
    if attrib_col < 2:
        attrib_col = 2
    arr = np.empty((0, 3), dtype="float32")
    size = data.shape[0]
    for i in range(size - 1):
        dis, angle = ha(data[i + 1:, :dim], data[i, :dim])
        halfvar = 0.5 * ((data[i + 1:, attrib_col] - data[i, attrib_col]) ** 2)
        dis.resize((dis.shape[0], 1))
        angle.resize((angle.shape[0], 1))
        halfvar.resize((halfvar.shape[0], 1))
        arr = np.append(arr, np.concatenate([dis, angle, halfvar], axis=1), axis=0)
    return arr


def calcHAVByVecs(vecs):
    distance = np.linalg.norm(vecs[:, :2], axis=1)
    angle = np.arctan2(vecs[:, 1], vecs[:, 0])
    angle[angle < 0] += np.pi
    var = 0.5 * (vecs[:, 2] ** 2)
    hav = np.zeros((len(vecs), 3))
    hav[:, 0] = distance
    hav[:, 1] = angle
    hav[:, 2] = var
    return hav


def calcHABVByVecs(vecs):
    distance = np.linalg.norm(vecs[:, :3], axis=1)
    azimuth = np.arctan2(vecs[:, 1], vecs[:, 0])
    azimuth[azimuth < 0] += np.pi
    norm = np.linalg.norm(vecs[:, :2], axis=1)
    dip = np.arctan2(vecs[:, 2], norm)
    dip[dip < 0] += np.pi
    var = 0.5 * (vecs[:, 3] ** 2)
    habv = np.zeros((len(vecs), 4))
    habv[:, 0] = distance
    habv[:, 1] = azimuth
    habv[:, 2] = dip
    habv[:, 3] = var
    return habv


def calcVecs(data, includeSelf=False, repeat=True):
    L = len(data)
    if repeat:
        vecs = []
        if includeSelf:
            for i in range(L):
                vec = data - data[i]
                vecs.append(vec)
        else:
            _indice = [i for i in range(L)]
            for i in range(L):
                indice = list(_indice)
                indice.pop(i)
                vec = data[indice] - data[i]
                vecs.append(vec)
        vecs = np.array(vecs)
        return vecs.reshape((vecs.shape[0] * vecs.shape[1], vecs.shape[2]))
    else:
        vecs = np.empty((0, data.shape[1]))
        for i in range(L - 1):
            vec = data[i + 1:] - data[i]
            vecs = np.append(vecs, vec, axis=0)
        return vecs


def kSplit(ndarray, k):
    size = ndarray.shape[0]
    step = size // k
    if size % k != 0:
        step += 1
    splits = []
    for i in range(k):
        begin = int(i * step)
        end = int((i + 1) * step)
        if i == k - 1:
            splits.append(ndarray[begin:])
        else:
            splits.append(ndarray[begin:end])
    return splits
