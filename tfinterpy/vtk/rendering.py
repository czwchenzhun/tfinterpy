# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 12:08
# @Author  : UCCU
import vtk
import tfinterpy.vtk.colorMap as CM
import numpy as np

def createPointsActor(x,y,z=0,scalar=None,scalarRange=None,colorMap=CM.BlueGreenOrange):
    '''
    Create vertices representing coordinate points based on the coordinate data.

    :param x: array_like, x coordinate data.
    :param y: array_like, y coordinate data.
    :param z: array_like, z coordinate data.
        If it is a two-dimensional space, z should be set to 0.
    :param scalar: array_like, properties.
    :param scalarRange: array_like, [scalarMin, scalarMax], properties' range.
    :param colorMap: list, refer to the colorMap in the vtk.colorMap module.
    :return: vtkActor object.
    '''
    points = vtk.vtkPoints()
    L=len(x)
    points.SetNumberOfPoints(L)
    if z==0:
        for i in range(L):
            points.SetPoint(i,x[i],y[i],0)
    else:
        for i in range(L):
            points.SetPoint(i,x[i],y[i],z[i])
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    scalarRange=None
    if scalar is not None:
        scalarArr = vtk.vtkFloatArray()
        scalarArr.SetName("scalar")
        scalarArr.SetNumberOfComponents(1)
        scalarArr.SetNumberOfValues(L)
        for i in range(L):
            scalarArr.SetTuple1(i,scalar[i])
        polyData.GetPointData().AddArray(scalarArr)
        polyData.GetPointData().SetActiveScalars("scalar")
        if scalarRange is None:
            scalarRange=[scalar.min(),scalar.max()]
    vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
    vertexGlyphFilter.AddInputData(polyData)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())
    if scalarRange is not None:
        mapper.SetScalarRange(*scalarRange)
        ctf = CM.getCTF(scalarRange[0], scalarRange[1], colorMap)
        ctf.SetDiscretize(True)
        ctf.Build()
        mapper.SetLookupTable(ctf)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)
    return actor

def createGridActor(nx,ny,nz,x,y,z,scalar=None,scalarRange=None,colorMap=CM.Rainbow):
    '''
    Create rectilinear grid based on the coordinate data.

    :param nx: integer, number of x coordinate.
    :param ny: integer, number of y coordinate.
    :param nz: integer, number of z coordinate.
    :param x: array_like, x coordinate data.
    :param y: array_like, y coordinate data.
    :param z: array_like, z coordinate data.
    :param scalar: array_like, properties.
    :param scalarRange: array_like, [scalarMin, scalarMax], properties' range.
    :param colorMap: list, refer to the colorMap in the vtk.colorMap module.
    :return: vtkActor object.
    '''
    rgrid = vtk.vtkRectilinearGrid()
    nz=1 if nz<=1 else nz
    z=[0] if nz==1 else z
    dim=[nx, ny, nz]
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
    for idx, value in enumerate(x):
        xcoords.SetValue(idx, value)
    for idx, value in enumerate(y):
        ycoords.SetValue(idx, value)
    for idx, value in enumerate(z):
        zcoords.SetValue(idx, value)
    rgrid.SetXCoordinates(xcoords)
    rgrid.SetYCoordinates(ycoords)
    rgrid.SetZCoordinates(zcoords)
    if scalar is not None:
        scalarArr = vtk.vtkFloatArray()
        scalarArr.SetName('scalar')
        scalarArr.SetNumberOfComponents(1)
        scalarArr.SetNumberOfValues(len(scalar))
        for idx,value in enumerate(scalar):
            scalarArr.SetValue(idx,value)
        rgrid.GetPointData().AddArray(scalarArr)
        rgrid.GetPointData().SetActiveScalars("scalar")
        if scalarRange is None:
            scalarRange = [np.percentile(scalar,1), np.percentile(scalar,99)]
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(rgrid)
    if scalarRange is not None:
        mapper.SetScalarRange(*scalarRange)
        ctf = CM.getCTF(scalarRange[0], scalarRange[1], colorMap)
        ctf.SetDiscretize(True)
        ctf.Build()
        mapper.SetLookupTable(ctf)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToGouraud()
    return actor

def rendering(actor):
    '''
    Create a new render window to render the actor.

    :param actor: vtkActor object.
    :return: None.
    '''
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    ren.SetBackground(0.447, 0.552, 0.756)
    renWin.SetSize(960, 680)
    iren.Initialize()
    ren.ResetCamera()
    renWin.Render()
    iren.Start()