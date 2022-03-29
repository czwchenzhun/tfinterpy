import numpy as np


def linear(h, slope, C0):
    return slope * h + C0


def power(h, scale, exp, C0):
    return scale * (h ** exp) + C0


def gaussian(h, C, a, C0):
    return C * (1.0 - np.exp(-(h ** 2.0) / (a * 4.0 / 7.0) ** 2.0)) + C0


def exponent(h, C, a, C0):
    return C * (1.0 - np.exp(-h / (a / 3.0))) + C0


def spherical(h, C, a, C0):
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
    func = VariogramModelMap.get(name)
    return func
