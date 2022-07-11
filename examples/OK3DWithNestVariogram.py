from tfinterpy.krige import OK
from tfinterpy.grid import Grid3D
from tfinterpy.variogram import calculateOmnidirectionalVariogram3D
from tfinterpy.gslib.fileUtils import readGslibPoints
from tfinterpy.vtk.rendering import createGridActor, rendering
import tfinterpy.vtk.colorMap as CM
from tfinterpy.vtk.fileUtils import saveVTKGrid
from tfinterpy.utils import calcVecs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filePath = "data/sample_data.gslib"
    N = 8
    df = readGslibPoints(filePath)# Read 3d points from gslib file.
    print(df.describe())
    samples = df[['x', 'y', 'z', 'porosity']].values

    # Create linear 3D grid.
    grid = Grid3D()
    grid.rectlinear((100, 100, 10), (samples[:, 0].min(), samples[:, 0].max()),
                    (samples[:, 1].min(), samples[:, 1].max()), (samples[:, 2].min(), samples[:, 2].max()))

    # Calculate a nested variogram function.
    nestVariogram, variogramBuilders = calculateOmnidirectionalVariogram3D(samples)
    for vb in variogramBuilders:
        plt.figure()
        vb.showVariogram()
        plt.show()

    exe = OK(samples, '3d')# Create a ok interpolator.
    # Perform interpolation of all points in the grid.
    grid.pro, grid.sigma = exe.execute(grid.points(), N, nestVariogram)

    print(exe.crossValidate(N, nestVariogram))# Perform leave-one-out validation and print result.
    print(exe.crossValidateKFold(10, N, nestVariogram))# Perform k-fold validation and print result.
    # Create an actor representing a rectilinear grid and use the interpolation results for color mapping.
    actor = createGridActor(*grid.dim, grid.x, grid.y, grid.z, grid.pro,
                            [samples[:, 3].min(), samples[:, 3].max()], CM.Rainbow)
    rendering(actor)# Rendering vtkActor.
    # Create an actor representing a rectilinear grid and use the kriging variances for color mapping.
    actor = createGridActor(*grid.dim, grid.x, grid.y, grid.z, grid.sigma, None, CM.Rainbow)
    rendering(actor)# Rendering vtkActor.
    saveVTKGrid('savedData/grid.vtk', actor.GetMapper().GetInput())# Save grid data(kriging variances) to vtk file.
