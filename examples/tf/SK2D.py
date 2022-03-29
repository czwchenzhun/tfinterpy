from tfinterpy.tf.krige import TFSK
from tfinterpy.grid import Grid2D
from examples.ncFileUtil import getSamples
from examples.plotUtils import *
import numpy as np
import tensorflow as tf
from tfinterpy.variogram import calculateDefaultVariogram2D
from tfinterpy.tf.variogramLayer import getVariogramLayer
import random

if __name__ == "__main__":
    random.seed(1)
    W, H = 100, 100
    M = 100
    offset = (400, 800)
    N = 8
    samples, lats, lons, ele = getSamples("../data/tatitlek_815_mhhw_2011.nc", 'lat', 'lon', 'Band1',
                                          offset, (W, H), M)
    grid = Grid2D()
    grid.rectlinear((W, H), (1, W), (1, H))

    vb = calculateDefaultVariogram2D(samples)
    plt.figure()
    vb.showVariogram()
    plt.show()
    vl = getVariogramLayer(vb)

    exe = TFSK(samples)
    with tf.device("/GPU:0"):
        grid.pro, grid.sigma = exe.execute(grid.points(), N, vl, 1000)

    print(exe.crossValidateKFold(10, N, vl))
    print(exe.crossValidate(N, vl))

    pro = grid.pro.reshape((grid.dim[1], grid.dim[0]))
    sigma = grid.sigma.reshape((grid.dim[1], grid.dim[0]))
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    img2d(ele, title="origin")
    plt.subplot(222)
    img2d(pro, title="estimate")
    absdiff = np.abs(pro - ele)
    plt.subplot(223)
    img2d(absdiff, title="error")
    print('result MAE:', np.mean(absdiff))
    plt.subplot(224)
    img2d(sigma, title="krige variance")
    plt.show()
