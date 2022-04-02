from tfinterpy.krige import OK
from examples.plotUtils import *
from tfinterpy.variogram import calculateOmnidirectionalVariogram2D
from tfinterpy.utils import calcVecs
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("data/AustraliaTemperature20210101.csv")
    samples = df[["LONGITUDE", "LATITUDE", "TEMP"]].values

    nv, vbs = calculateOmnidirectionalVariogram2D(samples, model=None)
    for vb in vbs:
        plt.figure()
        vb.showVariogram()
        plt.show()
    N = 8
    exe = OK(samples)
    print(exe.crossValidate(N, nv))
    print(exe.crossValidateKFold(10, N, nv))
