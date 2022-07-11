import numpy as np


def linear(h, slope, C0):
    '''
    Linear model.

    :param h: number or ndarray, distance.
    :param slope: number.
    :param C0: number, nugget.
    :return: number or ndarray.
    '''
    return slope * h + C0


def power(h, scale, exp, C0):
    '''
    Power model.

    :param h: number or ndarray, distance.
    :param scale: number.
    :param exp: number.
    :param C0: number, nugget.
    :return: number or ndarray.
    '''
    return scale * (h ** exp) + C0


def gaussian(h, C, a, C0):
    '''
    Gaussian model.

    :param h: number or ndarray, distance.
    :param C: number, sill.
    :param a: number, range.
    :param C0: number, nugget.
    :return: number or ndarray.
    '''
    return C * (1.0 - np.exp(-(h ** 2.0) / (a * 4.0 / 7.0) ** 2.0)) + C0


def exponent(h, C, a, C0):
    '''
    Exponent model.

    :param h: number or ndarray, distance.
    :param C: number, sill.
    :param a: number, range.
    :param C0: number, nugget.
    :return: number or ndarray.
    '''
    return C * (1.0 - np.exp(-h / (a / 3.0))) + C0


def spherical(h, C, a, C0):
    '''
    Spherical model.

    :param h: number or ndarray, distance.
    :param C: number, sill.
    :param a: number, range.
    :param C0: number, nugget.
    :return: number or ndarray.
    '''
    return np.piecewise(h, [h <= a, h > a],
                        [lambda x: C * ((3.0 * x) / (2.0 * a) - (x ** 3.0) / (2.0 * a ** 3.0)) + C0, C + C0, ])


VariogramModelMap = {
    "spherical": spherical,
    "gaussian": gaussian,
    "exponent": exponent,
    "linear": linear,
    "power": power,
}


def variogramModel(name):
    '''
    Returns the variogram function based on the name.

    :param name: str, variogram function name.
    :return: function.
    '''
    func = VariogramModelMap.get(name)
    return func
