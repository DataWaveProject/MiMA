#!/bin/bash

## This script removes any unecessary files created from the test 
## Run this before repeating the test if atmos_daily.nc and atmos_avg.nc already exist

echo "removing files from test directory"
ls -ltrh 
rm logfile.0000.out
rm time_stamp.out 
rm atmos_daily.nc*
rm atmos_avg.nc*
rm RESTART/*
rm ._mpp.nonrootpe.stdout
echo "done"
ls -ltrh
