## Single timestep test with restart

This test is a single timestep simulation of MiMA starting from a restart file. 
The output is checked against a baseline simulation. If successful, the output netcdf file 
should be identical.

## How to run this test

1. Compile MiMA in advance, either using cmake or mkmf.
2. Load all required modules for running (`ifort`, `icc`, `netcdf-c`, `netcdf-fortran`)
3. From within this directory, `. run_single_timestep_test.sh` 
 

