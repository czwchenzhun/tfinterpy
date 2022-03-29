import numpy as np


class Grid2D:
    def __init__(self):
        self.x = None
        self.y = None
        self.xran = None
        self.yran = None
        self._xy = None
        self.dim = [-1, -1]  # height and width

    def rectlinear(self, dim, xran, yran):
        self.dim = dim
        self.xran = xran
        self.yran = yran
        xstep = (xran[1] - xran[0]) / (dim[0] - 1)
        self.x = np.array([(i * xstep + xran[0]) for i in range(dim[0])])
        ystep = (yran[1] - yran[0]) / (dim[1] - 1)
        self.y = np.array([(i * ystep + yran[0]) for i in range(dim[1])])
        self._xy = None

    def setXCoords(self, xcoords):
        self.x = np.array(xcoords)
        self.dim[0] = len(self.x)
        self.xran = [self.x.min(), self.x.max()]
        self._xy = None

    def setYCoords(self, ycoords):
        self.y = np.array(ycoords)
        self.dim[1] = len(self.y)
        self.yran = [self.y.min(), self.y.max()]
        self._xy = None

    def points(self):
        if self._xy is None:
            self._xy = np.zeros((self.dim[1], self.dim[0], 2))
            for i in range(self.dim[1]):
                self._xy[i, :, 0] = self.x
            for i in range(self.dim[0]):
                self._xy[:, i, 1] = self.y
            self._xy.resize((self.dim[0] * self.dim[1], 2))
        return self._xy


class Grid3D:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.xran = None
        self.yran = None
        self.zran = None
        self._xyz = None
        self.dim = [-1, -1, -1]  # height, long, width

    def rectlinear(self, dim, xran, yran, zran):
        self.dim = dim
        self.xran = xran
        self.yran = yran
        self.zran = zran
        xstep = (xran[1] - xran[0]) / (dim[0] - 1)
        self.x = np.array([(i * xstep + xran[0]) for i in range(dim[0])])
        ystep = (yran[1] - yran[0]) / (dim[1] - 1)
        self.y = np.array([(i * ystep + yran[0]) for i in range(dim[1])])
        zstep = (zran[1] - zran[0]) / (dim[2] - 1)
        self.z = np.array([(i * zstep + zran[0]) for i in range(dim[2])])
        self._xyz = None

    def setXCoords(self, xcoords):
        self.x = np.array(xcoords)
        self.dim[0] = len(self.x)
        self.xran = [self.x.min(), self.x.max()]
        self._xyz = None

    def setYCoords(self, ycoords):
        self.y = np.array(ycoords)
        self.dim[1] = len(self.y)
        self.yran = [self.y.min(), self.y.max()]
        self._xyz = None

    def setZCoords(self, zcoords):
        self.z = np.array(zcoords)
        self.dim[2] = len(self.z)
        self.zran = [self.z.min(), self.z.max()]
        self._xyz = None

    def points(self):
        if self._xyz is None:
            self._xyz = np.zeros((self.dim[2], self.dim[1], self.dim[0], 3))
            for i in range(self.dim[2]):
                for j in range(self.dim[1]):
                    self._xyz[i, j, :, 0] = self.x
                for j in range(self.dim[0]):
                    self._xyz[i, :, j, 1] = self.y
                self._xyz[i, :, :, 2] = self.z[i]
            self._xyz.resize((self.dim[0] * self.dim[1] * self.dim[2], 3))
        return self._xyz
