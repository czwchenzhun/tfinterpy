import numpy as np


def linear(slope, C0, h):
    '''
    Linear model.

    :param slope: number.
    :param C0: number, nugget.
    :param h: number or ndarray, distance.
    :return: number or ndarray.
    '''
    return slope * h + C0


def power(scale, exp, C0, h):
    '''
    Power model.

    :param scale: number.
    :param exp: number.
    :param C0: number, nugget.
    :param h: number or ndarray, distance.
    :return: number or ndarray.
    '''
    return scale * (h ** exp) + C0

# nugget (C0), sill (C0+C), range(a)
# C should greater than 0, otherwise variogram curve will be a line.
# In the guarantee mae minimum, the bigger a and C, the better.

def gaussian(C, a, C0, h):
    '''
    Gaussian model.

    :param C: number, arch.
    :param a: number, range.
    :param C0: number, nugget.
    :param h: number or ndarray, distance.
    :return: number or ndarray.
    '''
    return C * (1.0 - np.exp(-(h ** 2.0) / (a * 4.0 / 7.0) ** 2.0)) + C0
    # return C * (1.0 - np.exp(-(h ** 2.0) / a ** 2.0)) + C0


def exponent(C, a, C0, h):
    '''
    Exponent model.

    :param C: number, arch.
    :param a: number, range.
    :param C0: number, nugget.
    :param h: number or ndarray, distance.
    :return: number or ndarray.
    '''
    return C * (1.0 - np.exp(-h / (a / 3.0))) + C0


def spherical(C, a, C0, h):
    '''
    Spherical model.

    :param C: number, arch.
    :param a: number, range.
    :param C0: number, nugget.
    :param h: number or ndarray, distance.
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
