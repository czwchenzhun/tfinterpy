from tfinterpy.tf.idw import TFIDW
from tfinterpy.grid import Grid2D
from examples.ncFileUtil import getSamples
from examples.plotUtils import *
import numpy as np
import tensorflow as tf
import random

print(tf.config.experimental.list_physical_devices('CPU'))
print(tf.config.experimental.list_physical_devices('GPU'))

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

    exe = TFIDW(samples)

    with tf.device("/GPU:0"):
        grid.pro = exe.execute(grid.points(), N)

    print(exe.crossValidate(N))
    print(exe.crossValidateKFold(10, N))

    pro = grid.pro.reshape((grid.dim[1], grid.dim[0]))
    plt.figure()
    plt.subplot(221)
    img2d(ele, title="origin")
    plt.subplot(222)
    img2d(pro, title="estimate")
    absdiff = np.abs(pro - ele)
    plt.subplot(223)
    img2d(absdiff, title="error")
    print('result MAE:', np.mean(absdiff))
    plt.show()
