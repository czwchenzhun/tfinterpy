import setuptools

with open("README.md","r",encoding="utf-8") as fi:
    long_description = fi.read()

setuptools.setup(
    name="TFInterpy",
    version="1.1.2",
    author="Zhiwen Chen",
    author_email="orchenz@qq.com",
    description="A high-performance spatial interpolation Python package. It contains IDW (Inverse Distance Weighted), SK (Simple Kriging), OK (Ordinary Kriging) algorithms, and visualization tools based on QT and VTK.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/czwchenzhun/tfinterpy.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
    ]
)