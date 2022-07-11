from tfinterpy.variogram import calculateDefaultVariogram3D
from tfinterpy.gslib.fileUtils import readGslibPoints
from tfinterpy.vtk.rendering import createGridActor, rendering
import tfinterpy.vtk.colorMap as CM
from tfinterpy.vtk.fileUtils import saveVTKGrid
# import vtk dependencies first
from tfinterpy.tf.variogramLayer import getVariogramLayer
from tfinterpy.tf.krige import TFOK
from tfinterpy.grid import Grid3D
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    print(tf.config.experimental.list_physical_devices('CPU'))  # Prints all CPU devices.
    print(tf.config.experimental.list_physical_devices('GPU'))  # Prints all GPU devices.

    filePath = "../data/sample_data.gslib"
    N = 8
    df = readGslibPoints(filePath)# Read 3d points from gslib file.
    print(df.describe())
    samples = df[['x', 'y', 'z', 'porosity']].values

    # Create linear 3D grid.
    grid = Grid3D()
    grid.rectlinear((100, 100, 10), (samples[:, 0].min(), samples[:, 0].max()),
                    (samples[:, 1].min(), samples[:, 1].max()), (samples[:, 2].min(), samples[:, 2].max()))

    # Calculate a default variogram function.
    vb = calculateDefaultVariogram3D(samples)
    plt.figure()
    vb.showVariogram()
    plt.show()
    vl = getVariogramLayer(vb)#Create variogram layer by variogram builder.

    exe = TFOK(samples, '3d')# Create a ok(tensorflow version) interpolator.

    # Specify the GPU to be used.
    with tf.device("/GPU:0"):
        # Perform interpolation of all points in the grid.
        grid.pro, grid.sigma = exe.execute(grid.points(), N, vl, 1000)

    # Create an actor representing a rectilinear grid and use the interpolation results for color mapping.
    actor = createGridActor(*grid.dim, grid.x, grid.y, grid.z, grid.pro,
                            [samples[:, 3].min(), samples[:, 3].max()], CM.Rainbow)
    rendering(actor)# Rendering vtkActor.
    # Create an actor representing a rectilinear grid and use the kriging variances for color mapping.
    actor = createGridActor(*grid.dim, grid.x, grid.y, grid.z, grid.sigma, None, CM.Rainbow)
    rendering(actor)# Rendering vtkActor.
    saveVTKGrid('../savedData/grid.vtk', actor.GetMapper().GetInput())# Save grid data(kriging variances) to vtk file.
