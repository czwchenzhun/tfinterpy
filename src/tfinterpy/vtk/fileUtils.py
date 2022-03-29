# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 19:53
# @Author  : UCCU

import vtk

def saveVTKPoints(filePath,polyData):
    writer=vtk.vtkPolyDataWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(polyData)
    writer.Write()

def saveVTKGrid(filePath,rectilinearGrid):
    writer=vtk.vtkRectilinearGridWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(rectilinearGrid)
    writer.SetWriteArrayMetaData(True)
    writer.Write()