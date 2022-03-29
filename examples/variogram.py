import matplotlib.pyplot as plt
from tfinterpy.variogram import VariogramBuilder
from tfinterpy.variogramExp import search2d
from examples.ncFileUtil import getSamples
from tfinterpy.utils import calcVecs, calcHAVByVecs
import numpy as np

if __name__ == "__main__":
    filePath = "data/tatitlek_815_mhhw_2011.nc"
    W, H = 200, 200
    M = 100
    offset = (100, 100)
    samples, lats, lons, ele = getSamples(filePath, 'lat', 'lon', 'Band1', offset, (W, H), M)
    vecs = calcVecs(samples, repeat=False)
    hav = calcHAVByVecs(vecs)
    lagNum = 20
    lag = hav[:, 0].max() / 2 / lagNum
    lagTole = lag * 0.5
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
    bandWidth = hav[:, 0].mean() / 10
    lags, _ = search2d(vecs[:, :2], hav[:, 2], lagNum, lag, lagTole, angle, angleTole, bandWidth)
    vb = VariogramBuilder(lags)
    plt.figure()
    vb.showVariogram()
    plt.show()
