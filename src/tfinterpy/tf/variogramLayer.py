import tensorflow as tf
from tensorflow.keras import layers


class LinearLayer(layers.Layer):
    def __init__(self, slope, C0):
        super(LinearLayer, self).__init__()
        self.slope = tf.constant(slope, dtype="float32")
        self.C0 = tf.constant(C0, dtype="float32")

    def call(self, h):
        return h * self.slope + self.C0


class PowerLayer(layers.Layer):
    def __init__(self, scale, exp, C0):
        super(PowerLayer, self).__init__()
        self.scale = tf.constant(scale, dtype="float32")
        self.exp = tf.constant(exp, dtype="float32")
        self.C0 = tf.constant(C0, dtype="float32")

    def call(self, h):
        return self.scale * (h ** self.exp) + self.C0


class GaussianLayer(layers.Layer):
    def __init__(self, C, a, C0):
        super(GaussianLayer, self).__init__()
        self.C = tf.constant(C, dtype="float32")
        self.a_2 = tf.constant((a * 4 / 7) ** 2, dtype="float32")
        self.C0 = tf.constant(C0, dtype="float32")

    def call(self, h):
        return self.C * (1.0 - tf.exp(-1.0 * (h ** 2.0) / self.a_2)) + self.C0


class ExponentLayer(layers.Layer):
    def __init__(self, C, a, C0):
        super(ExponentLayer, self).__init__()
        self.C = tf.constant(C, dtype="float32")
        self.a_ = tf.constant(a / 3.0, dtype="float32")
        self.C0 = tf.constant(C0, dtype="float32")

    def call(self, h):
        return self.C * (1.0 - tf.exp(-1.0 * h / self.a_)) + self.C0


class SphericalLayer(layers.Layer):
    def __init__(self, C, a, C0):
        super(SphericalLayer, self).__init__()
        self.C = tf.constant(C, dtype="float32")
        self.a = tf.constant(a, dtype="float32")
        self.a2 = tf.constant(2 * a, dtype="float32")
        self.a2_3 = tf.constant(self.a2 ** 3, dtype="float32")
        self.C0 = tf.constant(C0, dtype="float32")

    def call(self, h):
        condition = tf.greater(h, self.a)
        return tf.where(condition, self.C + self.C0, self.C * (h * 3.0 / self.a2 - h ** 3.0 / self.a2_3) + self.C0)


class NestVariogramLayer(layers.Layer):
    def __init__(self, variogramLayers, unitVectors):
        super(NestVariogramLayer, self).__init__()
        self.variogramLayers = variogramLayers
        self.unitVectors = unitVectors
        if len(self.unitVectors.shape) == 2:
            self.unitVectors = self.unitVectors.reshape((*self.unitVectors.shape, 1))

    def call(self, vec):
        total = 0
        for idx, variogramLayer in enumerate(self.variogramLayers):
            h = tf.abs(tf.matmul(vec, self.unitVectors[idx]))
            var = variogramLayer(h)
            total += var
        return total


VariogramLayerMap = {
    "gaussian": GaussianLayer,
    "exponent": ExponentLayer,
    "spherical": SphericalLayer,
    "linear": LinearLayer,
    "power": PowerLayer,
}


def getVariogramLayer(variogramBuilder):
    LayerClass = VariogramLayerMap[variogramBuilder.model]
    vl = LayerClass(*variogramBuilder.params)
    return vl


def getNestVariogramLayer(variogramBuilders, unitVectors):
    vls = []
    for vb in variogramBuilders:
        vl = getVariogramLayer(vb)
        vls.append(vl)
    return NestVariogramLayer(vls, unitVectors)
