import numpy as np


class Grid2D:
    '''
    Classes used to represent a two-dimensional grid.
    '''

    def __init__(self):
        self.x = None
        self.y = None
        self.xran = None
        self.yran = None
        self._xy = None
        self.dim = [-1, -1]  #width and height

    def rectlinear(self, dim, xran, yran):
        '''
        Initialize the two-dimensional grid according to the dimension, x range, y range.

        :param dim: array_like, array containing two integer. [width, height].
        :param xran: array_like, [xbegin, xend]. So the step of x is equal to (xend-xbegin)/(width-1).
        :param yran: array_like, [ybegin, yend].
        :return: None
        '''
        self.dim = dim
        self.xran = xran
        self.yran = yran
        xstep = (xran[1] - xran[0]) / (dim[0] - 1)
        self.x = np.array([(i * xstep + xran[0]) for i in range(dim[0])])
        ystep = (yran[1] - yran[0]) / (dim[1] - 1)
        self.y = np.array([(i * ystep + yran[0]) for i in range(dim[1])])
        self._xy = None

    def setXCoords(self, xcoords):
        '''
        Sets the x coordinate of the grid.

        :param xcoords: array_like, array containing all the coordinates along the x axis.
        :return: None
        '''
        self.x = np.array(xcoords)
        self.dim[0] = len(self.x)
        self.xran = [self.x.min(), self.x.max()]
        self._xy = None

    def setYCoords(self, ycoords):
        '''
        Sets the y coordinate of the grid.

        :param ycoords: array_like, array containing all the coordinates along the y axis.
        :return: None
        '''
        self.y = np.array(ycoords)
        self.dim[1] = len(self.y)
        self.yran = [self.y.min(), self.y.max()]
        self._xy = None

    def points(self):
        '''
        Returns an ndarray containing all coordinates on the grid,
        and the shape of the array is (width*height, 2).

        :return: ndarray
        '''
        if self._xy is None:
            self._xy = np.zeros((self.dim[1], self.dim[0], 2))
            for i in range(self.dim[1]):
                self._xy[i, :, 0] = self.x
            for i in range(self.dim[0]):
                self._xy[:, i, 1] = self.y
            self._xy.resize((self.dim[0] * self.dim[1], 2))
        return self._xy


class Grid3D:
    '''
    Classes used to represent a three-dimensional grid.
    '''

    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.xran = None
        self.yran = None
        self.zran = None
        self._xyz = None
        self.dim = [-1, -1, -1] # width(x), height(y), long(z)

    def rectlinear(self, dim, xran, yran, zran):
        '''
        Initialize the three-dimensional grid according to the dimension, x range, y range, xrange.

        :param dim: array_like, array containing two integer. [width, height].
        :param xran: array_like, [xbegin, xend]. So the step of x is equal to (xend-xbegin)/(width-1).
        :param yran: array_like, [ybegin, yend].
        :param zran: array_like, [zbegin, zend].
        :return: None
        '''
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
        '''
        Sets the x coordinate of the grid.

        :param xcoords: array_like, array containing all the coordinates along the x axis.
        :return: None
        '''
        self.x = np.array(xcoords)
        self.dim[0] = len(self.x)
        self.xran = [self.x.min(), self.x.max()]
        self._xyz = None

    def setYCoords(self, ycoords):
        '''
        Sets the y coordinate of the grid.

        :param ycoords: array_like, array containing all the coordinates along the y axis.
        :return: None
        '''
        self.y = np.array(ycoords)
        self.dim[1] = len(self.y)
        self.yran = [self.y.min(), self.y.max()]
        self._xyz = None

    def setZCoords(self, zcoords):
        '''
        Sets the z coordinate of the grid.

        :param zcoords: array_like, array containing all the coordinates along the z axis.
        :return: None
        '''
        self.z = np.array(zcoords)
        self.dim[2] = len(self.z)
        self.zran = [self.z.min(), self.z.max()]
        self._xyz = None

    def points(self):
        '''
        Returns an ndarray containing all coordinates on the grid,
        and the shape of the array is (width*height*long, 3).

        :return: ndarray
        '''
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
