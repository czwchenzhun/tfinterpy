from tfinterpy.utils import *
from tfinterpy.variogramExp import search2d, search3d
from tfinterpy.variogramModel import variogramModel, VariogramModelMap
from scipy.optimize import least_squares


def resident(params, x, y, variogram_function):
    error = variogram_function(x, *params) - y
    return error ** 2


def getX0AndBnds(h, y, variogram_model):
    if variogram_model == "linear":
        x0 = [(np.amax(y) - np.amin(y)) / 2 / (np.amax(h) - np.amin(h)), np.amin(y)]
        bnds = ([0.0, 0.0], [np.inf, np.amax(y)])
    elif variogram_model == "power":
        x0 = [(np.amax(y) - np.amin(y)) / (np.amax(h) - np.amin(h)), 1.0, np.amin(y)]
        bnds = ([0.0, 0.001, 0.0], [np.inf, 1.999, np.amax(y)])
    else:
        x0 = [(np.amax(y) - np.amin(y)) / 2, 0.5 * np.amax(h), np.amin(y)]
        bnds = ([0.0, 0.0, 0.0], [np.amax(y), np.amax(h), np.amax(y)])
    return x0, bnds


class VariogramBuilder:
    def __init__(self, lags, model="spherical"):
        self.model = model
        self.lags = np.array(lags)
        x0, bnds = getX0AndBnds(self.lags[:, 0], self.lags[:, 1], model)
        res = least_squares(resident, x0, bounds=bnds, loss="huber",
                            args=(self.lags[:, 0], self.lags[:, 1], variogramModel(model)))
        self.params = res.x
        self.mae = np.mean(res.fun)

    def showVariogram(self, axes=None):
        if axes is None:
            import matplotlib.pyplot as axes
        variogram = self.getVariogram()
        axes.scatter(self.lags[:, 0], self.lags[:, 1], alpha=0.5)
        max = np.max(self.lags[:, 0])
        X = np.arange(0, max, max / 100)
        Y = variogram(X)
        axes.plot(X, Y, alpha=0.5, color="red", label=self.model)

    def getVariogram(self):
        def variogram(h):
            return variogramModel(self.model)(h, *self.params)

        return variogram


class NestVariogram:
    def __init__(self, variograms, unitVectors):
        self.variograms = variograms
        self.unitVectors = unitVectors

    def __call__(self, vec):
        totalVar = 0
        for idx, unitVector in enumerate(self.unitVectors):
            h = np.abs(np.dot(vec, unitVector))
            var = self.variograms[idx](h)
            totalVar += var
        return totalVar


def calculateOmnidirectionalVariogram2D(vecs, vars, partitionNum=8, leastPairNum=10, lagNum=20, lagInterval=None,
                                        lagTole=None, bandWidth=None, model=None):
    azimuthStep = np.pi / partitionNum
    unitVectors = []
    azimuths = []
    for i in range(partitionNum):
        azimuth = (i + 0.5) * azimuthStep
        azimuths.append(azimuth)
        unitVectors.append([np.cos(azimuth), np.sin(azimuth)])
    unitVectors = np.array(unitVectors)
    azimuths = np.array(azimuths)
    norms = np.linalg.norm(vecs, axis=1)
    if bandWidth is None:
        bandWidth = norms.mean() / 2
    if lagInterval is None:
        lagInterval = norms.max() * 0.75 / lagNum
    if lagTole is None:
        lagTole = lagInterval / 2
    lagRanList = [[(i + 1) * lagInterval - lagTole, (i + 1) * lagInterval + lagTole] for i in range(lagNum)]
    bucket = [[[] for j in range(lagNum)] for i in range(partitionNum)]
    thetas = np.arctan2(vecs[:, 1], vecs[:, 0])
    thetas[thetas < 0] += np.pi
    indiceTheta = thetas // azimuthStep
    indiceTheta = indiceTheta.astype('int')
    for i in range(partitionNum):
        deltas = thetas - azimuths[i]
        bands = norms * deltas
        for j in range(lagNum):
            indice = np.where(
                ((indiceTheta == i) & (norms > lagRanList[j][0]) & (norms < lagRanList[j][1]) & (bands < bandWidth)))[0]
            bucket[i][j] = list(zip(norms[indice], vars[indice]))
    processedBucket = [[] for i in range(partitionNum)]
    searchedPairNumRecords = [[] for i in range(partitionNum)]
    for i in range(partitionNum):
        for j in range(lagNum):
            searchedPairNum = len(bucket[i][j])
            if searchedPairNum < 3:
                continue
            searchedPairNumRecords[i].append(searchedPairNum)
            lag = np.mean(bucket[i][j], axis=0)
            processedBucket[i].append(lag)
    lagsList = [None for i in range(partitionNum)]
    availableDir = []
    for i in range(partitionNum):
        if len(processedBucket[i]) > leastPairNum:
            lagsList[i] = processedBucket[i]
            availableDir.append(i)
    variogramBuilders = []
    if model is None:
        for lags in lagsList:
            if lags is None:
                continue
            minResident = float('+inf')
            best = None
            for key in VariogramModelMap:
                vb = VariogramBuilder(lags, key)
                if vb.mae < minResident:
                    minResident = vb.mae
                    best = vb
            variogramBuilders.append(best)
    else:
        for lags in lagsList:
            if lags is None:
                continue
            vb = VariogramBuilder(lags, model)
            variogramBuilders.append(vb)
    nestVariogram = NestVariogram([vb.getVariogram() for vb in variogramBuilders], unitVectors[availableDir])
    return nestVariogram, variogramBuilders


def calculateOmnidirectionalVariogram3D(vecs, vars, partitionNum=[6, 6], leastPairNum=10, lagNum=20, lagInterval=None,
                                        lagTole=None, bandWidth=None, model=None):
    if type(partitionNum) != list:
        partitionNum = [partitionNum, partitionNum]
    azimuthStep = np.pi / partitionNum[0]
    azimuths = [(i + 0.5) * azimuthStep for i in range(partitionNum[0])]
    dipStep = np.pi / partitionNum[1]
    dips = [(i + 0.5) * dipStep for i in range(partitionNum[1])]
    unitVectors = []
    for i in range(partitionNum[0]):
        a = azimuths[i]
        for j in range(partitionNum[1]):
            b = dips[j]
            unitVectors.append([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)])
    unitVectors = np.array(unitVectors)
    azimuths = np.array(azimuths)
    dips = np.array(dips)
    norms = np.linalg.norm(vecs, axis=1)
    if bandWidth is None:
        bandWidth = norms.mean() / 2
    if lagInterval is None:
        lagInterval = norms.max() * 0.75 / lagNum
    if lagTole is None:
        lagTole = lagInterval / 2
    lagRanList = [[(i + 1) * lagInterval - lagTole, (i + 1) * lagInterval + lagTole] for i in range(lagNum)]
    bucket = [[[[] for k in range(lagNum)] for j in range(partitionNum[1])] for i in range(partitionNum[0])]
    thetas1 = np.arctan2(vecs[:, 1], vecs[:, 0])
    thetas1[thetas1 < 0] += np.pi
    thetas2 = np.arctan2(vecs[:, 2], np.linalg.norm(vecs[:, :2], axis=1))
    thetas2[thetas2 < 0] += np.pi
    indiceTheta1 = thetas1 // azimuthStep
    indiceTheta1 = indiceTheta1.astype('int')
    indiceTheta2 = thetas2 // dipStep
    indiceTheta2 = indiceTheta2.astype('int')
    for i in range(partitionNum[0]):
        for j in range(partitionNum[1]):
            unitVector = unitVectors[i * partitionNum[1] + j]
            angles = np.arccos(np.dot(vecs, unitVector) / norms)
            indice = np.where(angles > np.pi / 2)[0]
            angles[indice] = np.pi - angles[indice]
            bands = norms * np.sin(angles)
            for k in range(lagNum):
                indice = np.where(
                    ((indiceTheta1 == i) & (indiceTheta2 == j) & (norms > lagRanList[k][0]) & (
                            norms < lagRanList[k][1]) & (bands < bandWidth)))[0]
                bucket[i][j][k] = list(zip(norms[indice], vars[indice]))
    processedBucket = [[[] for j in range(partitionNum[1])] for i in range(partitionNum[0])]
    searchedPairNumRecords = [[[] for j in range(partitionNum[1])] for i in range(partitionNum[0])]
    for i in range(partitionNum[0]):
        for j in range(partitionNum[1]):
            for k in range(lagNum):
                searchedPairNum = len(bucket[i][j][k])
                if searchedPairNum < 3:
                    continue
                searchedPairNumRecords[i][j].append(searchedPairNum)
                lag = np.mean(bucket[i][j][k], axis=0)
                processedBucket[i][j].append(lag)
    lagsList = [None for i in range(partitionNum[0] * partitionNum[1])]
    availableDir = []
    for i in range(partitionNum[0]):
        for j in range(partitionNum[1]):
            if len(processedBucket[i][j]) > leastPairNum:
                idx = i * partitionNum[1] + j
                lagsList[idx] = processedBucket[i][j]
                availableDir.append(idx)
    variogramBuilders = []
    if model is None:
        for lags in lagsList:
            if lags is None:
                continue
            minResident = float('+inf')
            best = None
            for key in VariogramModelMap:
                vb = VariogramBuilder(lags, key)
                if vb.mae < minResident:
                    minResident = vb.mae
                    best = vb
            variogramBuilders.append(best)
    else:
        for lags in lagsList:
            if lags is None:
                continue
            vb = VariogramBuilder(lags, model)
            variogramBuilders.append(vb)
    nestVariogram = NestVariogram([vb.getVariogram() for vb in variogramBuilders], unitVectors[availableDir])
    return nestVariogram, variogramBuilders


def calculateDefaultVariogram2D(samples, model='spherical'):
    vecs = calcVecs(samples, repeat=False)
    hav = calcHAVByVecs(vecs)
    lagNum = 20
    lagInterval = hav[:, 0].max() / 2 / lagNum
    lagTole = lagInterval * 0.5
    angles = hav[:, 1]
    buckets = []
    portionNum = 8
    portion = np.pi / portionNum
    for i in range(portionNum):
        count = np.where((angles > portion * i) & (angles < portion * (i + 1)))[0].shape[0]
        buckets.append(count)
    idx = np.argmax(buckets)
    angle = portion * idx
    angleTole = portion
    bandWidth = hav[:, 0].mean() / 2
    lags, _ = search2d(vecs[:, :2], hav[:, 2], lagNum, lagInterval, lagTole, angle, angleTole, bandWidth)
    if len(lags) < 3:
        lags = calcHV(samples)
    vb = VariogramBuilder(lags, model)
    return vb


def calculateDefaultVariogram3D(samples, model='spherical'):
    vecs = calcVecs(samples, repeat=False)
    habv = calcHABVByVecs(vecs)
    lagNum = 20
    lagInterval = habv[:, 0].max() / 2 / lagNum
    lagTole = lagInterval * 0.5
    azimuths = habv[:, 1]
    dips = habv[:, 2]
    buckets = []
    portionNum = 5
    portion = np.pi / portionNum
    for i in range(portionNum):
        for j in range(portionNum):
            count = np.where((azimuths > portion * i) & (azimuths < portion * (i + 1)) & (dips > portion * j) & (
                    dips < portion * (j + 1)))[0].shape[0]
            buckets.append(count)
    idx = np.argmax(buckets)
    azimuth = idx // portionNum * portion
    dip = idx % portionNum * portion
    angleTole = portion / 2
    bandWidth = habv[:, 0].mean() / 2
    lags, _ = search3d(vecs[:, :3], habv[:, 3], lagNum, lagInterval, lagTole, azimuth, dip, angleTole, bandWidth)
    if len(lags) < 3:
        lags = calcHV(samples)
    vb = VariogramBuilder(lags, model)
    return vb
