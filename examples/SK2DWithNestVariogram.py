from tfinterpy.krige import SK
from tfinterpy.grid import Grid2D
from tfinterpy.variogram import calculateDefaultVariogram2D, calculateOmnidirectionalVariogram2D
from tfinterpy.utils import calcVecs
from examples.ncFileUtil import getSamples
from examples.plotUtils import *
import numpy as np
import time

if __name__ == "__main__":
    filePath = "data/tatitlek_815_mhhw_2011.nc"
    W, H = 100, 100
    M = 100
    offset = (400, 800)
    N = 8
    samples, lats, lons, ele = getSamples(filePath, 'lat', 'lon', 'Band1', offset, (W, H), M)
    grid = Grid2D()
    grid.rectlinear((W, H), (1, W), (1, H))
    exe = SK(samples)

    nv, vbs = calculateOmnidirectionalVariogram2D(samples)
    indice = [i for i in range(0, len(nv.variograms), 2)]
    variograms = [nv.variograms[i] for i in indice]
    nv.variograms = variograms
    nv.unitVectors = nv.unitVectors[indice]

    grid.pro, grid.sigma = exe.execute(grid.points(), N, nv)

    print(exe.crossValidate(N, nv))
    print(exe.crossValidateKFold(10, N, nv))

    grid.pro.resize((grid.dim[1], grid.dim[0]))
    grid.sigma.resize((grid.dim[1], grid.dim[0]))
    ae = np.abs(grid.pro - ele)
    print("mae:", np.mean(ae))
    plt.subplot(221)
    img2d(ele, title="origin")
    plt.subplot(222)
    img2d(grid.pro, title="estimate")
    plt.subplot(223)
    img2d(ae, title="error")
    plt.subplot(224)
    img2d(grid.sigma, title="krige variance")
    plt.show()
