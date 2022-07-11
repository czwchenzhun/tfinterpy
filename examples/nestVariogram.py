import matplotlib.pyplot as plt
from tfinterpy.variogram import calculateOmnidirectionalVariogram2D
from examples.ncFileUtil import getSamples
from tfinterpy.utils import calcVecs

if __name__ == "__main__":
    filePath = "data/tatitlek_815_mhhw_2011.nc"
    W, H = 200, 200
    M = 100
    offset = (100, 100)

    # Get sample points and original elevation data from netcdf file
    # containing Digital Elevation Model data.
    samples, lats, lons, ele = getSamples(filePath, 'lat', 'lon', 'Band1', offset, (W, H), M)

    # Calculate the nested variogram function.
    nestVariogram, variogramBuilders = calculateOmnidirectionalVariogram2D(samples)
    for vb in variogramBuilders:
        plt.figure()
        vb.showVariogram()
        plt.show()
