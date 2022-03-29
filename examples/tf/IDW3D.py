import tfinterpy.vtk.colorMap as CM
from tfinterpy.vtk.rendering import createGridActor, rendering
from tfinterpy.vtk.fileUtils import saveVTKGrid
from tfinterpy.tf.idw import TFIDW
from tfinterpy.grid import Grid3D
from tfinterpy.gslib.fileUtils import readGslibPoints
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
    exe = TFIDW(samples, '3d')

    with tf.device("/GPU:0"):
        grid.pro = exe.execute(grid.points(), N)

    print(exe.crossValidate(N))
    print(exe.crossValidateKFold(10, N))

    actor = createGridActor(*grid.dim, grid.x, grid.y, grid.z, grid.pro, [samples[:, 3].min(), samples[:, 3].max()],
                            CM.Rainbow)
    rendering(actor)
    saveVTKGrid('../savedData/grid.vtk', actor.GetMapper().GetInput())
