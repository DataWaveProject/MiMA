# Model of an idealized Moist Atmosphere (MiMA) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.597136.svg)](https://doi.org/10.5281/zenodo.597136)
MiMA is an intermediate-complexity General Circulation Model with interactive water vapor and full radiation. It is published in

[M Jucker and EP Gerber, 2017: *Untangling the annual cycle of the tropical tropopause layer with an idealized moist model*, Journal of Climate 30, 7339-7358](http://dx.doi.org/10.1175/JCLI-D-17-0127.1)

for v1.x and 

[Garfinkel et al. (2020): *The building blocks of Northern Hemisphere wintertime stationary waves*, Journal of Climate](http://journals.ametsoc.org/doi/10.1175/JCLI-D-19-0181.1)

for v2.0.

Please see the documentation [online](http://mjucker.github.io/MiMA/) or [in the docs folder](docs/) for information about the model.

See the 30 second trailer on [YouTube](https://www.youtube.com/watch?v=8UfaFnGtCrk "Model of an idealized Moist Atmosphere (MiMA)"): 

[![MiMA thumbnail](https://img.youtube.com/vi/8UfaFnGtCrk/0.jpg)](https://www.youtube.com/watch?v=8UfaFnGtCrk "Model of an idealized Moist Atmosphere (MiMA)")

Although free to use under a GPLv3 license, we ask you to cite the relevant scientific work given in the documentation in any publications using this code.

## License

MiMA is distributed under a GNU GPLv3 license. That means you have permission to use, modify, and distribute the code, even for commercial use. However, you must make your code publicly available under the same license. See LICENSE.txt for more details.

AM2 is distributed under a GNU GPLv2 license. That means you have permission to use, modify, and distribute the code, even for commercial use. However, you must make your code publicly available under the same license.

RRTM/RRTMG: Copyright © 2002-2010, Atmospheric and Environmental Research, Inc. (AER, Inc.). This software
may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced
on each copy made. This model is provided as is without any express or implied warranties.


## Building for Machine Learning

### PyTorch
* Requires the fortran-pytorch library available [here](https://github.com/Cambridge-ICCS/fortran-pytorch-lib)
* Build with: 

    cmake -DTorch_DIR=<Path-to-venv>/lib/python3.11/site-packages/torch/share/cmake/Torc

* Build MiMA using:

### TensorFlow
* Requires the fortran-pytorch library available [here](https://github.com/Cambridge-ICCS/fortran-tf-lib)
* Requires TensorFlow C API:

    FILENAME=libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz
    wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}
    tar -C <PATH_TO_INSTALL_LOCATION> -xzf ${FILENAME}

* Build with:

    cmake .. -DTENSORFLOW_LOCATION=<PATH_TO_TF_C_API_INSTALL_LOCATION> -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_C_COMPILER=icc -DCMAKE_BUILD_TYPE=Release

### Forpy
* Requires a python venv with TensorFlow/PyTorch (and any other requirements) installed.

### MiMA
* Build with 

    cmake -DFLAG_FOR_ML ..

where `-DFLAG_FOR_ML` is one of `-DBUILD_PT_COUPLING`, `-DBUILD_TF_COUPLING`, `-DBUILD_FORPY_COUPLING`
