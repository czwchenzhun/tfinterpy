# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 20:05
# @Author  : UCCU

import vtk

CoolToWarm=[
    [0.0,[0,0,255]],
    [0.5,[255,255,255]],
    [1.0,[255,0,0]],
]

Rainbow=[
    [0.0,[0,0,255]],
    [0.2,[0,127,255]],
    [0.4,[0,255,0]],
    [0.6,[255,255,0]],
    [0.8,[255,165,0]],
    [1.0,[255,0,0]]
]

BlueGreenOrange=[
    [0.0,[205,228,248]],
    [0.33,[29,65,88]],
    [0.66,[221,218,153]],
    [1.0,[115,2,0]]
]

def getCTF(scalarMin,scalarMax,stops):
    '''
    Get vtk color transfer function by scalarRange and stops.

    :param scalarMin: number.
    :param scalarMax: number.
    :param stops: list, each stop represented by [percentage,[r,g,b]].
    :return: vtkDiscretizableColorTransferFunction object.
    '''
    ctf=vtk.vtkDiscretizableColorTransferFunction()
    total=scalarMax-scalarMin

    for stop in stops:
        ctf.AddRGBPoint(scalarMin+total*stop[0],stop[1][0]/255.0,stop[1][1]/255.0,stop[1][2]/255.0)
    return ctf