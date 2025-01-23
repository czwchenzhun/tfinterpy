## TFInterpy
TFInterpy is a Python package for spatial interpolation. A high-performance version of several interpolation algorithms is implemented based on TensorFlow. Including parallelizable IDW and Kriging algorithms. So far, tfinterpy is the **fastest open source Kriging** algorithm, which can reduce the operation time of large-scale interpolation tasks by an order of magnitude

## Link to our paper
[TFInterpy: A high-performance spatial interpolation Python package](https://www.sciencedirect.com/science/article/pii/S2352711022001479)
<br>
(https://doi.org/10.1016/j.softx.2022.101229)

### Performance comparison (unit: second)

| Grid size | GeostatsPy-OK | PyKrige-OK | TFInterpy-OK | TFInterpy-TFOK(GPU) | TFInterpy-TFOK(CPU) |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| 1x10<sup>4<sup/> | 23.977 | 1.258 | 0.828 | 2.070 | 0.979 |
| 1x10<sup>5<sup/> | 230.299 | 12.264 | 8.140 | 6.239 | 2.067 |
| 1x10<sup>6<sup/> | 2011.351 | 121.711 | 82.397 | 45.737 | 11.683 |
| 1x10<sup>7<sup/> | 2784.843 | 1250.980 | 849.974 | 452.567 | 112.331 |

### Screenshots
Snapshot of GUI tool.
![Snapshot of GUI tool](./figs/OK3D.jpg)

#### Requirements

**Minimum usage requirements:** Python 3+, Numpy, SciPy
**TensorFlow based algorithm:** TensorFlow 2
**GSLIB file support:** Pandas
**3D visualization:** VTK  
**GUI Tool:** PyQT5

-----

# Usage

#### Install tfinterpy
```
pip install tfinterpy
```

#### Then install dependencies

**Full dependencies** : (To avoid package version issues, the specific version numbers tested in Python3.9 are listed here)
```
pip install matplotlib==3.9.4
pip install numpy==2.0.2
pip install pandas==2.2.3
pip install PyQt5==5.15.11
pip install scipy==1.13.1
pip install tensorflow==2.18.0
pip install vtk==9.4.1
```

**Notice！** You may do not need to install all dependencies
- If you only need to use the most basic interpolation algorithm, install the following package. (see "examples/" for usage)
    ```
    pip install numpy==2.0.2
    pip install scipy==1.13.1
    ```
- If you need to use TensorFlow-based interpolation algorithms, you need to install tensorflow. (see "examples/tf" for usage)
    ```
    pip install tensorflow==2.18.0
    ```
    or (Use GPU for computing)
    ```
    pip install tensorflow-gpu==2.18.0
    ```
- If you need to use the built-in GUI tools (see "examples/gui" for usage) provided, please install full dependencies as above list.

netcdf4 also needs to be installed to **run the examples** in the examples folder:
```
pip install netCDF4==1.7.2
```

