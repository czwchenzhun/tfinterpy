from examples.ncFileUtil import getSamples
from tfinterpy.tf.krige import TFUK
from tfinterpy.tf.ukTrendFunc import Linear2D, Quadratic2D
from tfinterpy.grid import Grid2D
import tensorflow as tf
from examples.plotUtils import *
import numpy as np
from tfinterpy.variogram import calculateOmnidirectionalVariogram2D
from tfinterpy.tf.variogramLayer import getNestVariogramLayer
import random

if __name__ == "__main__":
    print(tf.config.experimental.list_physical_devices('CPU'))  # Prints all CPU devices.
    print(tf.config.experimental.list_physical_devices('GPU'))  # Prints all GPU devices.

    random.seed(1)
    W, H = 100, 100
    M = 100
    offset = (400, 800)
    N = 12

    # Get sample points and original elevation data from netcdf file
    # containing Digital Elevation Model data.
    samples, lats, lons, ele = getSamples("../data/tatitlek_815_mhhw_2011.nc", 'lat', 'lon', 'Band1',
                                          offset, (W, H), M)

    # Create linear 2D grid.
    grid = Grid2D()
    grid.rectlinear((W, H), (1, W), (1, H))

    # Calculate nested variation function.
    nestVariogram, variogramBuilders = calculateOmnidirectionalVariogram2D(samples)
    # Get nested variation function layer.
    nestVariogramLayer = getNestVariogramLayer(variogramBuilders, nestVariogram.unitVectors)

    exe = TFUK(samples)# Create a ok(tensorflow version) interpolator.

    # Specify the GPU to be used.
    # Perform interpolation of all points in the grid.
    grid.pro, grid.sigma = exe.execute(grid.points(), N, nestVariogramLayer, 5000, device='/CPU:0', trendFunc=Quadratic2D())

    # print(exe.crossValidateKFold(10, N, nestVariogramLayer))# Perform k-fold validation and print result.
    # print(exe.crossValidate(N, nestVariogramLayer))# Perform leave-one-out validation and print result.

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
