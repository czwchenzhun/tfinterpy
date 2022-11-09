from tfinterpy.gui.uiTool import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import numpy as np
import pandas as pd

from tfinterpy.variogramModel import VariogramModelMap
from tfinterpy.variogramExp import search2d, search3d
from tfinterpy.variogram import VariogramBuilder, calculateDefaultVariogram2D, calculateDefaultVariogram3D, \
    calculateOmnidirectionalVariogram2D, calculateOmnidirectionalVariogram3D
from tfinterpy.utils import calcVecs, calcHAVByVecs, calcHABVByVecs, calcHV
from tfinterpy.gui.canvas import Canvas
from tfinterpy.grid import Grid2D, Grid3D
from tfinterpy.vtk import colorMap
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import tensorflow as tf
from tfinterpy.idw import IDW
from tfinterpy.krige import SK, OK
from tfinterpy.tf.idw import TFIDW
from tfinterpy.tf.krige import TFSK, TFOK
from tfinterpy.tf.variogramLayer import getVariogramLayer, getNestVariogramLayer
from tfinterpy.gslib.fileUtils import readGslibPoints, saveGslibPoints, saveGslibGrid
from tfinterpy.vtk.fileUtils import saveVTKPoints, saveVTKGrid
from scipy.optimize import least_squares
import time

ColorMap = colorMap.Rainbow #default color map


class Tool(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Tool, self).__init__()
        self.setupUi(self)
        self.actorPS = None
        self.actorGrid = None
        self.actorAxes = None
        self.actorScalarBar = None
        self.initUi()
        self.connectSignals()
        self.data = None
        self.samples = None
        self.vl = None
        self.vb = None
        self.grid = None


    def initUi(self):
        self.vtkWidget = QVTKRenderWindowInteractor(self.tab_interp)
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.actorScalarBar = vtk.vtkScalarBarActor()
        self.actorScalarBar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.actorScalarBar.GetPositionCoordinate().SetValue(0.88, 0.3)
        self.actorScalarBar.SetWidth(0.1)
        self.actorScalarBar.SetHeight(0.5)
        self.actorScalarBar.SetOrientationToVertical()
        self.actorScalarBar.SetNumberOfLabels(8)
        self.actorScalarBar.SetTextPositionToSucceedScalarBar()
        self.actorScalarBar.SetVisibility(True)
        self.ren.SetBackground(0.447, 0.552, 0.756)

        self.iren.Start()

        vl = QVBoxLayout(self.tab_interp)
        vl.addWidget(self.vtkWidget)
        self.tab_interp.setLayout(vl)

        self.canvasVariogram = Canvas(1, 1, 100, self.tab_variogram)
        # self.axesVariogram = self.canvasVariogram.fig.add_subplot(111)
        self.vl_variogram.addWidget(self.canvasVariogram)

        self.canvasCV = Canvas(1, 1, 100, self.tab_cv)
        self.axesCV = self.canvasCV.fig.add_subplot(111)
        vl = QVBoxLayout(self.tab_cv)
        vl.addWidget(self.canvasCV)
        self.tab_cv.setLayout(vl)

        self.le_alpha.setValidator(QDoubleValidator())
        self.le_angle.setValidator(QDoubleValidator())
        self.le_angleDip.setValidator(QDoubleValidator())
        self.le_angleTole.setValidator(QDoubleValidator())
        self.le_bandWidth.setValidator(QDoubleValidator())
        self.le_lagInterval.setValidator(QDoubleValidator())
        self.le_lagTole.setValidator(QDoubleValidator())
        self.le_lagNum.setValidator(QIntValidator())
        self.le_bx.setValidator(QDoubleValidator())
        self.le_by.setValidator(QDoubleValidator())
        self.le_bz.setValidator(QDoubleValidator())
        self.le_ex.setValidator(QDoubleValidator())
        self.le_ey.setValidator(QDoubleValidator())
        self.le_ez.setValidator(QDoubleValidator())
        self.le_nx.setValidator(QIntValidator())
        self.le_ny.setValidator(QIntValidator())
        self.le_nz.setValidator(QIntValidator())

        self.comb_fieldZ.setEnabled(self.check_fieldZ.isChecked())
        self.gb_variogram.setVisible(False)
        self.gb_customVariogram.setVisible(False)
        self.setLagsSearchMode(False)
        self.comb_variogramModel.insertItems(0, [key for key in VariogramModelMap])
        self.comb_method.insertItems(0, ['IDW', 'SK', 'OK'])
        cpus = tf.config.experimental.list_physical_devices('CPU')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for cpu in cpus:
            name = cpu.name[cpu.name.find(':') + 1:]
            self.comb_device.addItem(name)
        for gpu in gpus:
            name = gpu.name[gpu.name.find(':') + 1:]
            self.comb_device.addItem(name)

    def dataChanged(self):
        fields = self.data.columns.tolist()
        self.comb_fieldX.clear()
        self.comb_fieldX.insertItems(0, fields)
        self.comb_fieldX.setCurrentIndex(0)
        self.comb_fieldY.clear()
        self.comb_fieldY.insertItems(0, fields)
        self.comb_fieldY.setCurrentIndex(1)
        self.comb_fieldZ.clear()
        self.comb_fieldZ.insertItems(0, fields)
        idx = 2
        if self.check_fieldZ.isChecked():
            self.comb_fieldY.setCurrentIndex(idx)
            idx += 1
        self.comb_fieldP.clear()
        self.comb_fieldP.insertItems(0, fields)
        self.comb_fieldP.setCurrentIndex(idx)

    def setAlphaVisibility(self, visible):
        self.label_alpha.setVisible(visible)
        self.le_alpha.setVisible(visible)

    def setLagsSearchMode(self, is3d):
        self.label_angleDip.setVisible(is3d)
        self.le_angleDip.setVisible(is3d)
        if is3d:
            self.label_angle.setText("azimuth angle")
        else:
            self.label_angle.setText("angle")

    def onloadFileClicked(self):
        dir = '.'
        curFileName = self.le_loadFile.text()
        if curFileName != "":
            dir = curFileName[:curFileName.rfind('/')]
        fileName, _ = QFileDialog.getOpenFileName(self, 'load file', dir, 'file (*.csv *.gslib)')
        if fileName == '':
            return
        self.le_loadFile.setText(fileName)
        if fileName.endswith('csv'):
            self.data = pd.read_csv(fileName)
        else:
            self.data = readGslibPoints(fileName)
        self.dataChanged()

    def setupSamples(self):
        if self.data is None:
            return
        cols = [self.comb_fieldX.currentText(), self.comb_fieldY.currentText(), self.comb_fieldP.currentText()]
        if self.check_fieldZ.isChecked():
            cols.insert(2, self.comb_fieldZ.currentText())
        self.samples = self.data[cols].values
        self.cols = cols

    def onCalcVariogramClicked(self):
        self.statusBar.showMessage("Processing...",10000)#show message in  status bar, duration time set to 10s.
        begin=time.perf_counter()
        self.setupSamples()
        if self.samples is None:
            return
        if self.rb_calcVariogramCustom.isChecked():
            if self.check_fieldZ.isChecked():
                try:
                    lagNum = eval(self.le_lagNum.text())
                    lagInterval = eval(self.le_lagInterval.text())
                    lagTole = eval(self.le_lagTole.text())
                    azimuth = eval(self.le_angle.text())
                    dip = eval(self.le_angleDip.text())
                    angleTole = eval(self.le_angleTole.text())
                    bandWidth = eval(self.le_bandWidth.text())
                except:
                    QMessageBox.critical(self, "Error", "Missing parameters!")
                    return
                vecs = calcVecs(self.samples, repeat=False)
                habv = calcHABVByVecs(vecs)
                lags, _ = search3d(vecs[:, :3], habv[:, 3], lagNum, lagInterval, lagTole, azimuth, dip, angleTole,
                                   bandWidth)
                if len(lags) < 3:
                    QMessageBox.critical(self, "Warning", "Lags number less than 3! Using all pairs.")
                    lags = calcHV(self.samples, 3)
                self.vb = VariogramBuilder(lags, self.variogramModel())
            else:
                try:
                    lagNum = eval(self.le_lagNum.text())
                    lagInterval = eval(self.le_lagInterval.text())
                    lagTole = eval(self.le_lagTole.text())
                    angle = eval(self.le_angle.text())
                    angleTole = eval(self.le_angleTole.text())
                    bandWidth = eval(self.le_bandWidth.text())
                except:
                    QMessageBox.critical(self, "Error", "Missing parameters!")
                    return
                vecs = calcVecs(self.samples, repeat=False)
                hav = calcHAVByVecs(vecs)
                lags, _ = search2d(vecs[:, :2], hav[:, 2], lagNum, lagInterval, lagTole, angle, angleTole, bandWidth)
                if len(lags) < 3:
                    QMessageBox.critical(self, "Warning", "Lags number less than 3! Using all pairs.")
                    lags = calcHV(self.samples)
                self.vb = VariogramBuilder(lags, self.variogramModel())
        elif self.rb_calcVariogramDefault.isChecked():
            if self.check_fieldZ.isChecked():
                self.vb = calculateDefaultVariogram3D(self.samples, self.variogramModel())
            else:
                self.vb = calculateDefaultVariogram2D(self.samples, self.variogramModel())
        else:
            if self.check_fieldZ.isChecked():
                self.vb = calculateOmnidirectionalVariogram3D(self.samples)
            else:
                self.vb = calculateOmnidirectionalVariogram2D(self.samples)
        if type(self.vb) == tuple:
            self.canvasVariogram.fig.clear()
            nestVariogram, variogramBuilders = self.vb
            unitVectors = nestVariogram.unitVectors
            col = 3
            row = int(np.ceil(len(unitVectors) / col))
            for idx, vb in enumerate(variogramBuilders):
                if len(unitVectors[0]) == 3:
                    dip = np.arcsin(unitVectors[idx][2])
                    if np.cos(dip) == 0.0:
                        azimuth = 0.0
                    else:
                        azimuth = np.arccos(unitVectors[idx][0] / np.cos(dip))
                    text = 'azimuth: {:.4f}, dip: {:.4f}'.format(azimuth, dip)
                else:
                    azimuth = np.arccos(unitVectors[idx][0])
                    text = 'azimuth: {:.4f}'.format(azimuth)
                axes = self.canvasVariogram.fig.add_subplot(row, col, idx + 1)
                vb.showVariogram(axes)
                axes.legend()
                axes.set_title(text)
                axes.set_xlabel('distance')  # ,loc='right')
                axes.set_ylabel('variance')  # ,loc='top')
            self.canvasVariogram.fig.subplots_adjust(wspace=0.5, hspace=0.7)
            self.canvasVariogram.draw()
            self.canvasVariogram.show()
        else:
            self.canvasVariogram.fig.clear()
            self.axesVariogram = self.canvasVariogram.fig.add_subplot(111, )
            self.axesVariogram.cla()
            self.vb.showVariogram(self.axesVariogram)
            self.axesVariogram.set(title='Variogram', xlabel='distance', ylabel='variance')
            self.canvasVariogram.draw()
            self.canvasVariogram.show()
        self.tabWidget.setCurrentIndex(1)
        self.rb_variogramCalced.setEnabled(True)
        self.rb_variogramCalced.setChecked(True)
        end = time.perf_counter()
        t = end - begin
        self.statusBar.showMessage("Time consumed(/s): {:.4f}".format(t),10000)

    def onShowPointsClicked(self):
        if self.actorPS is not None:
            self.ren.RemoveActor(self.actorPS)
        if self.actorAxes is not None:
            self.ren.RemoveActor(self.actorAxes)
        self.setupSamples()
        if self.samples is None:
            return
        self.le_bx.setText(str(self.samples[:, 0].min()))
        self.le_by.setText(str(self.samples[:, 1].min()))
        self.le_ex.setText(str(self.samples[:, 0].max()))
        self.le_ey.setText(str(self.samples[:, 1].max()))
        if self.check_fieldZ.isChecked():
            self.le_bz.setText(str(self.samples[:, 2].min()))
            self.le_ez.setText(str(self.samples[:, 2].max()))

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(self.samples.shape[0])
        scalar = vtk.vtkFloatArray()
        scalar.SetName("scalar")
        scalar.SetNumberOfComponents(1)
        scalar.SetNumberOfValues(self.samples.shape[0])
        if self.check_fieldZ.isChecked():
            bounds = [self.samples[:, 0].min(), self.samples[:, 0].max(), self.samples[:, 1].min(),
                      self.samples[:, 1].max(), self.samples[:, 2].min(), self.samples[:, 2].max()]
            scalarRange = [self.samples[:, 3].min(), self.samples[:, 3].max()]
            for idx, point in enumerate(self.samples):
                points.SetPoint(idx, point[:3])
                scalar.SetTuple1(idx, point[3])
        else:
            bounds = [self.samples[:, 0].min(), self.samples[:, 0].max(), self.samples[:, 1].min(),
                      self.samples[:, 1].max(), 0, 0]
            scalarRange = [self.samples[:, 2].min(), self.samples[:, 2].max()]
            for idx, point in enumerate(self.samples):
                points.SetPoint(idx, point[0], point[1], 0)
                scalar.SetTuple1(idx, point[2])
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.GetPointData().AddArray(scalar)
        polyData.GetPointData().SetActiveScalars("scalar")
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.AddInputData(polyData)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())
        mapper.SetScalarRange(*scalarRange)
        ctf = colorMap.getCTF(scalarRange[0], scalarRange[1], ColorMap)
        ctf.SetDiscretize(True)
        ctf.Build()
        mapper.SetLookupTable(ctf)
        self.actorScalarBar.SetLookupTable(ctf)
        self.ren.AddActor2D(self.actorScalarBar)
        self.ctf = ctf
        self.scalarRange = scalarRange

        self.actorPS = vtk.vtkActor()
        self.actorPS.SetMapper(mapper)
        self.actorPS.GetProperty().SetPointSize(5)
        self.ren.AddActor(self.actorPS)

        self.actorAxes = vtk.vtkCubeAxesActor()
        self.actorAxes.SetBounds(bounds)
        self.actorAxes.SetCamera(self.ren.GetActiveCamera())
        self.axesMarker = vtk.vtkOrientationMarkerWidget()
        self.axesMarker.SetOutlineColor(0, 0, 0)
        self.axesMarker.SetOrientationMarker(self.actorAxes)
        self.axesMarker.SetCurrentRenderer(self.ren)
        self.axesMarker.SetInteractor(self.iren)
        self.axesMarker.SetEnabled(1)
        self.axesMarker.InteractiveOff()
        self.ren.AddActor(self.actorAxes)

        self.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        # self.axesPoints.cla()
        # mappable=self.axesPoints.scatter(self.samples[:, 0], self.samples[:, 1],c=self.samples[:,2],cmap='rainbow')
        # self.canvasPoints.figure.colorbar(mappable,ax=self.axesPoints)
        # self.canvasPoints.draw()
        # self.canvasPoints.show()
        self.tabWidget.setCurrentIndex(0)

    def onZCheckStateChanged(self, state):
        self.le_bz.setEnabled(state)
        self.le_ez.setEnabled(state)
        self.le_nz.setEnabled(state)
        if state:
            pass
        else:
            pass

    def cv(self):
        mode = '3d' if self.check_fieldZ.isChecked() else '2d'
        N = self.sb_neighSize.value()
        method = self.comb_method.currentText()
        if method == 'IDW':
            alpha = eval(self.le_alpha.text())
            if self.gb_useTF.isChecked():
                device = self.comb_device.currentText()
                exe = TFIDW(self.samples, mode)
                with tf.device('/' + device):
                    _, _, error = exe.crossValidate(N, alpha)
            else:
                exe = IDW(self.samples, mode)
                _, _, error = exe.crossValidate(N, alpha)
        else:
            if self.rb_variogramNone.isChecked():
                vb = None
            elif self.rb_variogramDefault.isChecked():
                vb = calculateDefaultVariogram2D(self.samples, self.variogramModel())
            else:
                vb = self.vb
            if self.gb_useTF.isChecked():
                Method = TFSK if method == 'SK' else TFOK
            else:
                Method = SK if method == 'SK' else OK
            exe = Method(self.samples, mode)
            if self.gb_useTF.isChecked():
                device = self.comb_device.currentText()
                with tf.device('/' + device):
                    if vb is None:
                        variogram = None
                    elif type(vb) == tuple:
                        variogram = getNestVariogramLayer(vb[1], vb[0].unitVectors)
                    else:
                        variogram = getVariogramLayer(vb)
                    _, _, error = exe.crossValidate(N, variogram)
            else:
                if vb is None:
                    variogram = None
                elif type(vb) == tuple:
                    variogram = vb[0]
                else:
                    variogram = vb.getVariogram()
                _, _, error = exe.crossValidate(N, variogram)
        idx = 2 if mode == '2d' else 3
        estimate = error + self.samples[:, idx]

        def resident(params, x, y):
            return (y - (params[0] * x + params[1])) ** 2

        res = least_squares(resident, [1, 0], args=(self.samples[:, idx], estimate))
        a, b = res.x

        self.axesCV.cla()
        self.axesCV.scatter(self.samples[:, idx], estimate, label="pairs")
        vran = [self.samples[:, idx].min(), self.samples[:, idx].max()]
        X = np.arange(vran[0], vran[1], (vran[1] - vran[0]) / 50.0)
        self.axesCV.plot(X, X, label="1:1 line", color="orange", linestyle='-')
        self.axesCV.plot(X, X * a + b, label="best linear line fit", color="red", linestyle='--')
        self.axesCV.set(title="Cross Validation", xlabel="sample values", ylabel="estimate values")
        self.axesCV.legend()
        self.canvasCV.draw()
        self.canvasCV.show()
        self.tabWidget.setCurrentIndex(2)

    def interp(self):
        mode = '3d' if self.check_fieldZ.isChecked() else '2d'
        try:
            bx = eval(self.le_bx.text())
            by = eval(self.le_by.text())
            ex = eval(self.le_ex.text())
            ey = eval(self.le_ey.text())
            nx = eval(self.le_nx.text())
            ny = eval(self.le_ny.text())
            if mode == '3d':
                bz = eval(self.le_bz.text())
                ez = eval(self.le_ez.text())
                nz = eval(self.le_nz.text())
                grid = Grid3D()
                grid.rectlinear([nx, ny, nz], [bx, ex], [by, ey], [bz, ez])
            else:
                grid = Grid2D()
                grid.rectlinear([nx, ny], [bx, ex], [by, ey])
        except:
            QMessageBox.critical(self, "Error", "Missing parameters!")
            return False
        N = self.sb_neighSize.value()
        method = self.comb_method.currentText()
        if method == 'IDW':
            alpha = eval(self.le_alpha.text())
            if self.gb_useTF.isChecked():
                device = self.comb_device.currentText()
                exe = TFIDW(self.samples, mode)
                with tf.device('/' + device):
                    grid.properties = exe.execute(grid.points(), N, alpha)
            else:
                exe = IDW(self.samples, mode)
                grid.properties = exe.execute(grid.points(), N, alpha)
            grid.sigmas = None
        else:
            if self.rb_variogramNone.isChecked():
                vb = None
            elif self.rb_variogramDefault.isChecked():
                vb = calculateDefaultVariogram2D(self.samples, self.variogramModel())
            else:
                vb = self.vb
            if self.gb_useTF.isChecked():
                Method = TFSK if method == 'SK' else TFOK
            else:
                Method = SK if method == 'SK' else OK
            exe = Method(self.samples, mode)
            if self.gb_useTF.isChecked():
                device = self.comb_device.currentText()
                with tf.device('/' + device):
                    if vb is None:
                        variogram = None
                    elif type(vb) == tuple:
                        variogram = getNestVariogramLayer(vb[1], vb[0].unitVectors)
                    else:
                        variogram = getVariogramLayer(vb)
                    grid.properties, grid.sigmas = exe.execute(grid.points(), N, variogram)
            else:
                if vb is None:
                    variogram = None
                elif type(vb) == tuple:
                    variogram = vb[0]
                else:
                    variogram = vb.getVariogram()
                grid.properties, grid.sigmas = exe.execute(grid.points(), N, variogram)
        self.grid = grid
        return True

    def showGrid(self):
        if self.actorGrid is not None:
            self.ren.RemoveActor(self.actorGrid)
        if self.actorAxes is not None:
            self.ren.RemoveActor(self.actorAxes)
        mode = '2d' if self.grid.__class__ == Grid2D else '3d'
        rgrid = vtk.vtkRectilinearGrid()
        dim = self.grid.dim if mode == '3d' else [*self.grid.dim, 1]
        rgrid.SetDimensions(dim)
        xcoords = vtk.vtkFloatArray()
        xcoords.SetNumberOfValues(dim[0])
        xcoords.SetNumberOfComponents(1)
        ycoords = vtk.vtkFloatArray()
        ycoords.SetNumberOfValues(dim[1])
        ycoords.SetNumberOfComponents(1)
        zcoords = vtk.vtkFloatArray()
        zcoords.SetNumberOfValues(dim[2])
        zcoords.SetNumberOfComponents(1)
        for idx, value in enumerate(self.grid.x):
            xcoords.SetValue(idx, value)
        for idx, value in enumerate(self.grid.y):
            ycoords.SetValue(idx, value)
        z = [0] if mode == '2d' else self.grid.z
        for idx, value in enumerate(z):
            zcoords.SetValue(idx, value)
        rgrid.SetXCoordinates(xcoords)
        rgrid.SetYCoordinates(ycoords)
        rgrid.SetZCoordinates(zcoords)

        scalar = vtk.vtkFloatArray()
        scalar.SetName('properties')
        scalar.SetNumberOfComponents(1)
        scalar.SetNumberOfValues(len(self.grid.properties))
        for idx, value in enumerate(self.grid.properties):
            scalar.SetValue(idx, value)
        rgrid.GetPointData().AddArray(scalar)
        if self.grid.sigmas is not None:
            scalar = vtk.vtkFloatArray()
            scalar.SetName('sigmas')
            scalar.SetNumberOfComponents(1)
            scalar.SetNumberOfValues(len(self.grid.sigmas))
            for idx, value in enumerate(self.grid.sigmas):
                scalar.SetValue(idx, value)
            rgrid.GetPointData().AddArray(scalar)

        rgrid.GetPointData().SetActiveScalars("properties")

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(rgrid)
        mapper.SetScalarRange(*self.scalarRange)
        mapper.SetLookupTable(self.ctf)

        self.actorGrid = vtk.vtkActor()
        self.actorGrid.SetMapper(mapper)
        self.actorGrid.GetProperty().SetInterpolationToGouraud()
        self.ren.AddActor(self.actorGrid)

        bounds = [*self.grid.xran, *self.grid.yran]
        if mode == '2d':
            bounds.extend([0, 0])
        else:
            bounds.extend([*self.grid.zran])
        self.actorAxes = vtk.vtkCubeAxesActor()
        self.actorAxes.SetBounds(bounds)
        self.actorAxes.SetCamera(self.ren.GetActiveCamera())
        self.axesMarker = vtk.vtkOrientationMarkerWidget()
        self.axesMarker.SetOutlineColor(0, 0, 0)
        self.axesMarker.SetOrientationMarker(self.actorAxes)
        self.axesMarker.SetCurrentRenderer(self.ren)
        self.axesMarker.SetInteractor(self.iren)
        self.axesMarker.SetEnabled(1)
        self.axesMarker.InteractiveOff()
        self.ren.AddActor(self.actorAxes)

        self.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        self.tabWidget.setCurrentIndex(0)

    def onExecuteInterpClicked(self):
        self.setupSamples()
        if self.samples is None:
            QMessageBox.critical(self, "Error", "No sample data!")
            return
        self.statusBar.showMessage("Processing...", 5000)
        begin = time.perf_counter()
        if self.rb_interp.isChecked():
            if not self.interp():
                return
            self.showGrid()
        else:
            self.cv()
        end = time.perf_counter()
        t = end - begin
        self.statusBar.showMessage("Time consumed(/s): {:.4f}".format(t),5000)

    def variogramModel(self):
        return self.comb_variogramModel.currentText()

    def connectSignals(self):
        self.pb_loadFile.clicked.connect(self.onloadFileClicked)
        self.check_fieldZ.stateChanged.connect(lambda state: self.comb_fieldZ.setEnabled(state))
        self.check_fieldZ.stateChanged.connect(self.onZCheckStateChanged)
        self.check_fieldZ.stateChanged.connect(self.setLagsSearchMode)
        self.pb_calcVariogram.clicked.connect(self.onCalcVariogramClicked)
        self.pb_showPoints.clicked.connect(self.onShowPointsClicked)
        self.rb_interp.toggled.connect(lambda state: self.gb_grid.setVisible(state))
        self.pb_executeInterp.clicked.connect(self.onExecuteInterpClicked)
        self.comb_method.currentTextChanged.connect(lambda text: self.setAlphaVisibility(text == 'IDW'))
        self.comb_method.currentTextChanged.connect(lambda text: self.gb_variogram.setVisible(text != 'IDW'))
        self.rb_calcVariogramCustom.toggled.connect(lambda state: self.gb_customVariogram.setVisible(state))
        self.pb_savePs.clicked.connect(self.onSavePsClicked)
        self.pb_saveGrid.clicked.connect(self.onSaveGridClicked)

    def onSavePsClicked(self):
        self.setupSamples()
        if self.samples is None:
            QMessageBox.critical(self, "Error", "No sample data!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save points, gslib or vtk", '', 'vtk(*.vtk);;gslib(*.gslib)')
        if path == '':
            return
        if path.endswith('vtk'):
            saveVTKPoints(path, self.actorPS.GetMapper().GetInput())
        else:
            saveGslibPoints(path, self.cols, self.samples)

    def onSaveGridClicked(self):
        if self.grid is None:
            QMessageBox.critical(self, "Error", "No grid!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save grid, gslib or vtk", '', 'vtk(*.vtk);;gslib(*.gslib)')
        if path == '':
            return
        if path.endswith('vtk'):
            saveVTKGrid(path, self.actorGrid.GetMapper().GetInput())
        else:
            begin = [self.grid.xran[0], self.grid.yran[0]]
            step = [(self.grid.xran[1] - self.grid.xran[0]) / (self.grid.dim[0] - 1),
                    (self.grid.yran[1] - self.grid.yran[0]) / (self.grid.dim[1] - 1)]
            if self.grid.__class__ == Grid3D:
                begin.append(self.grid.zran[0])
                step.append((self.grid.zran[1] - self.grid.zran[0]) / (self.grid.dim[2] - 1))
            colNames = ['properties']
            if self.grid.sigmas is not None:
                colNames.append('sigmas')
                data = np.concatenate([self.grid.properties, self.grid.sigmas], axis=1)
            else:
                data = self.grid.properties
            saveGslibGrid(path, self.grid.dim, begin, step, colNames, data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    exe = Tool()
    exe.showMaximized()
    sys.exit(app.exec_())
