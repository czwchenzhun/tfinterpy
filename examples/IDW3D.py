from tfinterpy.idw import IDW
from tfinterpy.grid import Grid3D
from tfinterpy.gslib.fileUtils import readGslibPoints
import tfinterpy.vtk.colorMap as CM
from tfinterpy.vtk.rendering import createGridActor, rendering
from tfinterpy.vtk.fileUtils import saveVTKGrid

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
    exe = IDW(samples, '3d')# Create a idw interpolator.

    grid.pro = exe.execute(grid.points(), N)# Perform interpolation of all points in the grid.
    print(exe.crossValidate(N))# Perform leave-one-out validation and print result.
    print(exe.crossValidateKFold(10, N))# Perform k-fold validation and print result.

    # Create an actor representing a rectilinear grid and use the interpolation results for color mapping.
    actor = createGridActor(*grid.dim, grid.x, grid.y, grid.z, grid.pro,
                            [samples[:, 3].min(), samples[:, 3].max()], CM.Rainbow)
    rendering(actor)# Rendering vtkActor.
    saveVTKGrid('savedData/grid.vtk', actor.GetMapper().GetInput())# Save grid data to vtk file.
