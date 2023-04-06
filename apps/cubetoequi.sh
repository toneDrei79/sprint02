#!/bin/bash

# arg1: path of 360-view image

/usr/local/cuda/bin/nvcc cubetoequi.cu -w `pkg-config opencv4 --cflags --libs` cubetoequi.cpp -o apps/cubetoequi
./apps/cubetoequi $1