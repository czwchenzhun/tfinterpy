import netCDF4 as nc
import random
import numpy as np
import os
import os.path as op


def getNCFileList(dirPath):
    '''
    Returns all netcdf files' path under the specified directory.

    :param dirPath: str, directory.
    :return: list, files' path.
    '''
    ls = []
    if not op.exists(dirPath):
        return ls
    for file in os.listdir(dirPath):
        if file[-3:] == '.nc':
            ls.append(file)
    return ls


def peakNCWH(filePath, wk="lon", hk="lat"):
    '''
    Get the width and height information contained in the nc file.

    :param filePath: str.
    :param wk: str, the key corresponding to the width.
    :param hk: str, the key corresponding to the height.
    :return: tuple, (width, height).
        If the file does not contain width and height information,
        return (None,None).
    '''
    if not op.exists(filePath):
        return None, None
    try:
        dem = nc.Dataset(filePath)
    except:
        print("netcdf 文件读取异常")
        return None, None
    W = dem.dimensions[wk].size
    H = dem.dimensions[hk].size
    dem.close()
    return W, H


def peakNCInfo(filePath):
    '''
     Get the key and corresponding shape information contained in the nc file.

    :param filePath: str.
    :return: tuple, (keys,info).
        keys is list containing all key in nc file.
        info is a string that records the shape of the variables.
    '''
    if not op.exists(filePath):
        return None, None
    try:
        dem = nc.Dataset(filePath)
    except:
        print("netcdf 文件读取异常")
        return None, None
    info = ''
    keys = []
    for key in dem.variables.keys():
        if dem.variables[key].shape == ():
            continue
        keys.append(key)
        info += "%s shape: %s\n" % (key, str(dem.variables[key].shape))
    dem.close()
    return keys, info


def getSamples(filePathOrData, X, Y, Z, begin=(0, 0), sub=(200, 200), N=10):
    '''
    :param filePath: str.
    :param X: str, key of lats.
    :param Y: str, key of lons.
    :param Z: str, key of ele.
    :param begin: tuple, the begining index of the submap.
    :param sub: tuple, the size of the submap.
    :param N: integer, number of samples in X and Y directions.
    :return: tuple, (samples,lats,lons,ele).
        samples is the sampling point obtained by random sampling within the submap.
        lats is latitude of submap.
        lons is longitude of submap.
        ele is elevation of submap.
    '''
    if type(filePathOrData) == type(''):
        dem = nc.Dataset(filePathOrData)
        data = dem.variables
    else:
        data = filePathOrData
    by, bx = begin
    suby, subx = sub
    lats = data[X][bx:bx + subx]  # 纬度值
    lons = data[Y][by:by + suby]  # 经度值
    ele = data[Z][bx:bx + subx, by:by + suby]  # 海拔值
    if type(filePathOrData) == type(''):
        dem.close()  #
    subx, suby = sub
    samples = []
    c = 0
    while c < N:
        x = random.randint(0, subx - 1)
        y = random.randint(0, suby - 1)
        if [x, y, ele[y, x]] not in samples:
            samples.append([x, y, ele[y, x]])
            c += 1
    samples = np.array(samples, dtype="float32")
    return samples, lats, lons, ele
