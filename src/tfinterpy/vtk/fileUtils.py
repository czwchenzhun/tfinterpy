# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 19:53
# @Author  : UCCU

import vtk

def saveVTKPoints(filePath,polyData):
    '''
    Save vtkPoints to file.

    :param filePath: str.
    :param polyData: vtkPolyData object.
    :return: None.
    '''
    writer=vtk.vtkPolyDataWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(polyData)
    writer.Write()

def saveVTKGrid(filePath,rectilinearGrid):
    '''
    Save vtkRectilinearGrid to file.

    :param filePath: str.
    :param rectilinearGrid: vtkRectilinearGrid object.
    :return: None.
    '''
    writer=vtk.vtkRectilinearGridWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(rectilinearGrid)
    writer.SetWriteArrayMetaData(True)
    writer.Write()