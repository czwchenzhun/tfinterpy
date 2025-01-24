from tfinterpy.krige import SK
from tfinterpy.grid import Grid2D
from tfinterpy.variogram import calculateDefaultVariogram2D
from ncFileUtil import getSamples
from plotUtils import *
import numpy as np
import random

if __name__ == "__main__":
    random.seed(1)
    filePath = "data/tatitlek_815_mhhw_2011.nc"
    W, H = 100, 100
    M = 100
    offset = (400, 800)
    N = 8

    # Get sample points and original elevation data from netcdf file
    # containing Digital Elevation Model data.
    samples, lats, lons, ele = getSamples(filePath, 'lat', 'lon', 'Band1', offset, (W, H), M)

    # Create linear 2D grid.
    grid = Grid2D()
    grid.rectlinear((W, H), (1, W), (1, H))

    # Calculate a default variogram function.
    vb = calculateDefaultVariogram2D(samples)
    exe = SK(samples)# Create a sk interpolator.
    # Perform interpolation of all points in the grid.
    grid.pro, grid.sigma = exe.execute(grid.points(), N, vb.getVariogram())

    print(exe.crossValidate(N, vb.getVariogram()))# Perform leave-one-out validation and print result.
    print(exe.crossValidateKFold(10, N, vb.getVariogram()))# Perform k-fold validation and print result.

    grid.pro.resize((grid.dim[1], grid.dim[0]))# Reshape properties as 2d ndarray.
    grid.sigma.resize((grid.dim[1], grid.dim[0]))# Reshape kriging variances as 2d ndarray.
    ae = np.abs(grid.pro - ele)
    print("mae:", np.mean(ae))
    plt.subplot(221)
    img2d(ele, title="origin")# Plotting original elevation data.
    plt.subplot(222)
    img2d(grid.pro, title="estimate")# Plotting interpolation results.
    plt.subplot(223)
    img2d(ae, title="error")# Plotting absolute errors.
    plt.subplot(224)
    img2d(grid.sigma, title="krige variance")# Plotting kriging variances.
    plt.show()
