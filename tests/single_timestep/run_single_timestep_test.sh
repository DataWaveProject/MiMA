#!/bin/bash

# Minimal runscript for single timestep
ulimit -s unlimited

# MiMA must be compiled and all modules must be loaded before running
N_PROCS=8
mppncombine="../../build/mppnccombine"
executable="../../build/mima.x"

echo "Single timestep run"
echo "Executable in ${executable}"
echo "Run with $N_PROCS processors"

ulimit -s unlimited

echo "run MiMA"
[ ! -d RESTART ] && mkdir RESTART

srun --ntasks $N_PROCS $executable

echo "Run complete. Postprocess with mppncombine in ${mppncombine}"

$mppncombine -r atmos_daily.nc atmos_daily.nc.????
$mppncombine -r atmos_avg.nc atmos_avg.nc.????

echo "Done"

echo "Test is the output the same as the baseline files?"
cdfdiff atmos_daily.nc  atmos_daily_base.nc 
cdfdiff atmos_avg.nc  atmos_avg_base.nc


