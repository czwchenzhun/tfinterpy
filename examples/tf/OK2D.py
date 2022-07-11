from tfinterpy.tf.krige import TFOK
from tfinterpy.grid import Grid2D
import tensorflow as tf
from examples.ncFileUtil import getSamples
from examples.plotUtils import *
import numpy as np
from tfinterpy.variogram import calculateDefaultVariogram2D
from tfinterpy.tf.variogramLayer import getVariogramLayer
import random

if __name__ == "__main__":
    print(tf.config.experimental.list_physical_devices('CPU'))  # Prints all CPU devices.
    print(tf.config.experimental.list_physical_devices('GPU'))  # Prints all GPU devices.

    random.seed(1)
    W, H = 200, 200
    M = 100
    offset = (100, 300)
    N = 8

    # Get sample points and original elevation data from netcdf file
    # containing Digital Elevation Model data.
    samples, lats, lons, ele = getSamples("../data/tatitlek_815_mhhw_2011.nc", 'lat', 'lon', 'Band1',
                                          offset, (W, H), M)

    # Create linear 2D grid.
    grid = Grid2D()
    grid.rectlinear((W, H), (1, W), (1, H))

    # Calculate a default variogram function.
    vb = calculateDefaultVariogram2D(samples)
    plt.figure()
    vb.showVariogram()
    plt.show()
    vl = getVariogramLayer(vb)#Create variogram layer by variogram builder.

    exe = TFOK(samples)# Create a ok(tensorflow version) interpolator.

    # Specify the GPU to be used.
    with tf.device("/GPU:0"):
        # Perform interpolation of all points in the grid.
        grid.pro, grid.sigma = exe.execute(grid.points(), N, vl, 1000)

    print(exe.crossValidateKFold(10, N, vl))# Perform k-fold validation and print result.
    print(exe.crossValidate(N, vl))# Perform leave-one-out validation and print result.

    pro = grid.pro.reshape((grid.dim[1], grid.dim[0]))# Reshape properties as 2d ndarray.
    sigma = grid.sigma.reshape((grid.dim[1], grid.dim[0]))# Reshape kriging variances as 2d ndarray.
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    img2d(ele, title="origin")# Plotting original elevation data.
    plt.subplot(222)
    img2d(pro, title="estimate")# Plotting interpolation results.
    absdiff = np.abs(pro - ele)
    plt.subplot(223)
    img2d(absdiff, title="error")# Plotting absolute errors.
    print('result MAE:', np.mean(absdiff))
    plt.subplot(224)
    img2d(sigma, title="krige variance")# Plotting kriging variances.
    plt.show()
