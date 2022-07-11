import numpy as np

EPS = 1e-16

def search2d(vecs, vars, lagNum, lagInterval, lagTole, angle, angleTole, bandWidth):
    '''
    Search for lags in two dimensions.

    :param vecs: ndarray, vectors.
    :param vars: ndarray, semivariance.
    :param lagNum: integer, number of lags to search.
    :param lagInterval: number, interval lags' distance.
    :param lagTole: number, tolerance of lag's distance.
    :param angle: number, search direction, angle range is [0, PI].
    :param angleTole: number, angle tolerance.
    :param bandWidth: number, bandwidth used during search.
    :return: tuple, (processedLags, lags).
        processedLags is the mean-processed lags.
        lags is original searched lags.
    '''
    unitVector = np.array([np.cos(angle), np.sin(angle)])
    return searchLags(vecs, vars, lagNum, lagInterval, lagTole, unitVector, angleTole, bandWidth)


def search3d(vecs, vars, lagNum, lagInterval, lagTole, azimuth, dip, angleTole, bandWidth):
    '''

    :param vecs: ndarray, vectors.
    :param vars: ndarray, semivariance.
    :param lagNum: integer, number of lags to search.
    :param lagInterval: number, interval lags' distance.
    :param lagTole: number, tolerance of lag's distance.
    :param azimuth: number, azimuth of search direction, the range of azimuth is [0, PI].
    :param dip: number, dip of search direction, the range of dip is [0, PI].
    :param angleTole: number, angle tolerance.
    :param bandWidth: number, bandwidth used during search.
    :return: tuple, (processedLags, lags).
        processedLags is the mean-processed lags.
        lags is original searched lags.
    '''
    unitVector = np.array([np.cos(azimuth) * np.cos(dip), np.sin(azimuth) * np.cos(dip), np.sin(dip)])
    return searchLags(vecs, vars, lagNum, lagInterval, lagTole, unitVector, angleTole, bandWidth)


def searchLags(vecs, vars, lagNum, lagInterval, lagTole, unitVector, angleTole, bandWidth):
    '''
    Search lags.

    :param vecs: ndarray, vectors.
    :param vars: ndarray, semivariance.
    :param lagNum: integer, number of lags to search.
    :param lagInterval: number, interval lags' distance.
    :param lagTole: number, tolerance of lag's distance.
    :param unitVector: array_like, unit vector of search direction.
    :param angleTole: number, angle tolerance.
    :param bandWidth: number, bandwidth used during search.
    :return: tuple, (processedLags, lags).
        processedLags is the mean-processed lags.
        lags is original searched lags.
    '''
    norms = np.linalg.norm(vecs, axis=1) + EPS
    thetas = np.arccos(np.dot(vecs, unitVector) / norms)
    indice = np.where(thetas > np.pi / 2)[0]
    thetas[indice] = np.pi - thetas[indice]
    # filter with angle
    remain = np.where(thetas <= angleTole)[0]
    norms = norms[remain]
    thetas = thetas[remain]
    vars = vars[remain]
    # filter with bandwith
    bands = norms * np.sin(thetas)
    remain = np.where(bands <= bandWidth)[0]
    norms = norms[remain]
    vars = vars[remain]
    # filter with lag range
    lagRanList = []
    for i in range(1, lagNum + 1):
        lagRanList.append([lagInterval * i - lagTole, lagInterval * i + lagTole])
    lags = [[] for i in range(lagNum)]
    for i in range(len(norms)):
        h = norms[i]
        for j in range(len(lagRanList)):
            if h > lagRanList[j][0] and h < lagRanList[j][1]:
                lags[j].append([h, vars[i]])
                break
    processedLags = []
    for ls in lags:
        if len(ls)==0:
            continue
        mean = np.mean(ls, axis=0)
        if np.isnan(mean).any():
            continue
        processedLags.append(mean)
    return processedLags, lags
