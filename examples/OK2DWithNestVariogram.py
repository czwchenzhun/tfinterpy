from tfinterpy.krige import OK
from tfinterpy.grid import Grid2D
from examples.ncFileUtil import getSamples
from examples.plotUtils import *
from tfinterpy.variogram import calculateOmnidirectionalVariogram2D
from tfinterpy.utils import calcVecs
import numpy as np
import random

if __name__ == "__main__":
    random.seed(1)
    filePath = "data/tatitlek_815_mhhw_2011.nc"
    W, H = 100, 100
    M = 100
    offset = (400, 800)
    N = 8
    samples, lats, lons, ele = getSamples(filePath, 'lat', 'lon', 'Band1', offset, (W, H), M)
    grid = Grid2D()
    grid.rectlinear((W, H), (1, W), (1, H))

    vecs = calcVecs(samples, repeat=False)
    vecs[:, 2] = 0.5 * vecs[:, 2] ** 2
    nv, vbs = calculateOmnidirectionalVariogram2D(vecs[:, :2], vecs[:, 2], model=None)
    for vb in vbs:
        plt.figure()
        vb.showVariogram()
        plt.show()

    exe = OK(samples)
    grid.pro, grid.sigma = exe.execute(grid.points(), N, nv)

    print(exe.crossValidate(N, nv))
    print(exe.crossValidateKFold(10, N, nv))

    grid.pro.resize((grid.dim[1], grid.dim[0]))
    grid.sigma.resize((grid.dim[1], grid.dim[0]))

    ae = np.abs(grid.pro - ele)
    print("mae:", np.mean(ae))
    plt.subplot(221)
    img2d(ele, "origin")
    plt.subplot(222)
    img2d(grid.pro, "estimate")
    plt.subplot(223)
    img2d(ae, "error")
    plt.subplot(224)
    img2d(grid.sigma, "krige variance")
    plt.show()
