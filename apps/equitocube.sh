#!/bin/bash

# arg1: path of 360-view image

/usr/local/cuda/bin/nvcc equitocube.cu -w `pkg-config opencv4 --cflags --libs` equitocube.cpp -o apps/equitocube
./apps/equitocube $1