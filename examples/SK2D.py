from tfinterpy.krige import SK
from tfinterpy.grid import Grid2D
from tfinterpy.variogram import calculateDefaultVariogram2D
from examples.ncFileUtil import getSamples
from examples.plotUtils import *
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
    vb = calculateDefaultVariogram2D(samples)

    exe = SK(samples)
    grid.pro, grid.sigma = exe.execute(grid.points(), N, vb.getVariogram())

    print(exe.crossValidate(N, vb.getVariogram()))
    print(exe.crossValidateKFold(10, N, vb.getVariogram()))

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
