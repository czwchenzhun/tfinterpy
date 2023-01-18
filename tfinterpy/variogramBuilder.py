import numpy as np
from scipy.optimize import least_squares
from functools import partial
from tfinterpy.settings import dtype
from tfinterpy.variogramModel import variogramModel, VariogramModelMap


def getX0AndBnds(h, y, variogram_model):
    '''
    Calculate the initial value and range of parameters(nugget, sill, range, ...)
    according to the variogram function model.

    :param h: array_like, distance.
    :param y: array_like, semivariance.
    :param variogram_model: str, indicates the variogram function model.
    :return: tuple, (x0,bnds). x0 represents the initial value of the parameters,
        and bnds represents the range of the parameters.
    '''
    if variogram_model == "linear":
        x0 = [(np.amax(y) - np.amin(y)) / 2 / (np.amax(h) - np.amin(h)), np.amin(y)]
        bnds = ([0.0, 0.0], [+np.inf, np.amax(y)])
    elif variogram_model == "power":
        x0 = [(np.amax(y) - np.amin(y)) / (np.amax(h) - np.amin(h)), 1.0, np.amin(y)]
        bnds = ([0.0, 0.001, 0.0], [+np.inf, 1.999, np.amax(y)])
    else:
        hmin, hmax = np.amin(h), np.amax(h)
        hran = hmax - hmin
        ymin, ymax = np.amin(y), np.amax(y)
        yran = ymax - ymin
        x0 = [yran / 2, 0.5 * hmax, ymin]
        amin = hmax / 50 if hmax / 50 > 1e-9 else 1e-9
        # bnds = ([yran*0.1, amin, 0.0], [yran, hmax, yran])
        bnds = ([0, amin, 0.0], [yran, hmax, ymin + yran])
    return x0, bnds


def resident(params, x, y, variogram_function):
    '''
    This function is used to calculate resident.

    :param params: tuple, represents the parameters passed to the variogram function.
    :param x: number or ndarray, distance.
    :param y: number or ndarray, semivariance.
    :param variogram_function: function, variogram function.
    :return: number or ndarray, resident=( variogram(x,params[0],...,params[n])-y )^2
    '''
    # error = variogram_function(x, *params) - y
    error = variogram_function(*params, x) - y
    return error ** 2


class VariogramBuilder:
    '''
    VariogramBuilder class is used to build variogram function.
    '''

    def __init__(self, lags, model="spherical", lagNumBeforeAvg=None):
        '''
        The VariogramBuilder is constructed with lags and string indicating the model.

        :param lags: array_like, a lag is [distance,semivariance].
        :param model: str, indicates the variogram function model.
        '''
        self.lags = np.array(lags, dtype=dtype)
        self.model = model
        w = 0
        h_1 = 1 / (self.lags[:, 0] ** 0.1)
        sumh_1 = np.sum(h_1)
        w = h_1 / sumh_1
        if not lagNumBeforeAvg is None:
            w += lagNumBeforeAvg / np.sum(lagNumBeforeAvg)

        def func(params, x, y, variogram_function):
            error = variogram_function(*params, x) - y
            return w * (error ** 2)

        x0, bnds = getX0AndBnds(self.lags[:, 0], self.lags[:, 1], model)
        res = least_squares(func, x0, bounds=bnds, loss="huber",
                            args=(self.lags[:, 0], self.lags[:, 1], variogramModel(model)))
        self.params = res.x
        self.mae = np.mean(res.fun)

    def showVariogram(self, axes=None):
        '''
        Plot variogram.

        :param axes: axes, if axes is set to None, plot on a new axes.
        :return: None.
        '''
        if axes is None:
            import matplotlib.pyplot as axes
        variogram = self.getVariogram()
        axes.scatter(self.lags[:, 0], self.lags[:, 1], alpha=0.5)
        max = np.max(self.lags[:, 0])
        X = np.arange(0, max, max / 1000)
        Y = variogram(X)
        axes.plot(X, Y, alpha=0.5, color="red", label=self.model)

    def getVariogram(self):
        '''
        Get variogram function(closure function).
        :return: function.
        '''
        func = variogramModel(self.model)
        pfunc = partial(func, *self.params)
        return pfunc
