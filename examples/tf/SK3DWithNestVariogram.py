from tfinterpy.variogram import calculateOmnidirectionalVariogram3D
from tfinterpy.gslib.fileUtils import readGslibPoints
import tfinterpy.vtk.colorMap as CM
from tfinterpy.vtk.rendering import createGridActor, rendering
from tfinterpy.vtk.fileUtils import saveVTKGrid
# import vtk dependencies first
from tfinterpy.tf.variogramLayer import getNestVariogramLayer
from tfinterpy.tf.krige import TFSK
from tfinterpy.grid import Grid3D
from tfinterpy.utils import calcVecs
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == "__main__":
    filePath = "../data/sample_data.gslib"
    N = 8
    df = readGslibPoints(filePath)
    print(df.describe())
    samples = df[['x', 'y', 'z', 'porosity']].values
    grid = Grid3D()
    grid.rectlinear((100, 100, 10), (samples[:, 0].min(), samples[:, 0].max()),
                    (samples[:, 1].min(), samples[:, 1].max()), (samples[:, 2].min(), samples[:, 2].max()))

    nestVariogram, variogramBuilders = calculateOmnidirectionalVariogram3D(samples, [5, 5])
    for vb in variogramBuilders:
        plt.figure()
        vb.showVariogram()
        plt.show()
    vl = getNestVariogramLayer(variogramBuilders, nestVariogram.unitVectors)

    exe = TFSK(samples, '3d')
    with tf.device("/GPU:0"):
        grid.pro, grid.sigma = exe.execute(grid.points(), N, vl, 10000)

    print(exe.crossValidateKFold(10, N, vl))
    print(exe.crossValidate(N, vl))

    actor = createGridActor(*grid.dim, grid.x, grid.y, grid.z, grid.pro, [samples[:, 3].min(), samples[:, 3].max()],
                            CM.Rainbow)
    rendering(actor)
    actor = createGridActor(*grid.dim, grid.x, grid.y, grid.z, grid.sigma, None, CM.Rainbow)
    rendering(actor)
    saveVTKGrid('../savedData/grid.vtk', actor.GetMapper().GetInput())

