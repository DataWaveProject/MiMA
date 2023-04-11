## Single timestep test

This test is a single timestep simulation of MiMA. The output is checked against 
a baseline simulation. If successful, the output netcdf file should be identical.

## How to run this test

1. Compile MiMA in advance, either using cmake or mkmf.
2. Load all required modules for running (`ifort`, `icc`, `netcdf-c`, `netcdf-fortran`)
3. From within this directory, `. run_single_timestep_test.sh` 
You may need to specify the path to the mima executable with `-x` and the mppncombine 
executable with `-c`. The default setting assumes they are both in `../../build`. You can 
also choose the number of processors with `-p`. The default is 4.
Example:
`. run_single_timestep_test.sh -p 4 -x ../exp/exec/mima.x -c ../bin/mppnccombine`


