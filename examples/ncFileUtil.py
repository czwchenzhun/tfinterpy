import netCDF4 as nc
import random
import numpy as np
import os
import os.path as op


def getNCFileList(dirPath):
    ls = []
    if not op.exists(dirPath):
        return ls
    for file in os.listdir(dirPath):
        if file[-3:] == '.nc':
            ls.append(file)
    return ls


def peakNCWH(filePath, wk="lon", hk="lat"):
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
    :param filePath: netcdf文件路径
    :param X: lats（纬度的字段）
    :param Y: lons（经度的字段）
    :param Z: ele（海拔的字段）
    :param begin: 采样子图的起点位置
    :param sub: 采样子图的size
    :param per: X和Y方向的采样数目
    :return: samples（子图内随机采样得到的采样点）,lats（子图的纬度），lons（子图的经度）,ele（子图的海拔）
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
