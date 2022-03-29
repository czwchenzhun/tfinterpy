from tfinterpy.krige import OK
from examples.plotUtils import *
from tfinterpy.variogram import calculateOmnidirectionalVariogram2D
from tfinterpy.utils import calcVecs
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("data/AustraliaTemperature20210101.csv")
    samples = df[["LONGITUDE", "LATITUDE", "TEMP"]].values
    vecs = calcVecs(samples, repeat=False)
    vecs[:, 2] = 0.5 * vecs[:, 2] ** 2
    nv, vbs = calculateOmnidirectionalVariogram2D(vecs[:, :2], vecs[:, 2], model=None)
    for vb in vbs:
        plt.figure()
        vb.showVariogram()
        plt.show()
    N = 8
    exe = OK(samples)
    print(exe.crossValidate(N, nv))
    print(exe.crossValidateKFold(10, N, nv))
