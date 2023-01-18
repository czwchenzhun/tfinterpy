import tensorflow as tf
from tensorflow.keras import layers

class TrendFunc(layers.Layer):
    def __init__(self):
        super(TrendFunc, self).__init__()
        self.items = []

    @tf.function(jit_compile=True)
    def call(self, loc):
        res = tf.concat([[]], 0)
        for f in self.items:
            res = tf.concat([res, [f(loc)]], 0)
        return res

    def __len__(self):
        return len(self.items)


class Linear2D(TrendFunc):
    '''
    x+y
    m(x) = a1*x + a2*y
    '''

    def __init__(self):
        super(Linear2D, self).__init__()
        self.items.append(lambda p: p[0])
        self.items.append(lambda p: p[1])


class Linear3D(TrendFunc):
    '''
    x+y+z
    m(x) = a1*x + a2*y + a3*z
    '''

    def __init__(self):
        super(Linear3D, self).__init__()
        self.items.append(lambda p: p[0])
        self.items.append(lambda p: p[1])
        self.items.append(lambda p: p[2])


class Quadratic2D(TrendFunc):
    '''
    (x+y)^2
    m(x) = a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y
    '''

    def __init__(self):
        super(Quadratic2D, self).__init__()
        self.items.append(lambda p: p[0])
        self.items.append(lambda p: p[1])
        self.items.append(lambda p: p[0] * p[0])
        self.items.append(lambda p: p[1] * p[1])
        self.items.append(lambda p: p[0] * p[1])



class Quadratic3D(TrendFunc):
    '''
    (x+y+z)^2
    m(x) = x^2 + y^2 + z^2 + 2*x*y + 2*x*z + 2*y*z
    '''

    def __init__(self):
        super(Quadratic3D, self).__init__()
        self.items.append(lambda p: p[0] * p[0])
        self.items.append(lambda p: p[1] * p[1])
        self.items.append(lambda p: p[2] * p[2])
        self.items.append(lambda p: p[0] * p[1])
        self.items.append(lambda p: p[0] * p[2])
        self.items.append(lambda p: p[1] * p[2])