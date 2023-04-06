/*
    convert cubem map image into equiretangular
*/

#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"


#define PI 3.1415

__global__ void cubetoequi(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int sqr)
{   
    // global dst coordinates
    const int2 g_coord = {
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y
    };

    // polar angles
    float theta = (1.-float(g_coord.x))/cols * 2*PI;
    float phi = float(g_coord.y)/rows * PI;

    // polar coordinates
    int3 coord = {
        cos(theta) * sin(phi),
        sin(theta) * sin(phi),
        cos(phi)
    };

    float maximum = max(abs(coord.x), abs(coord.y), abs(coord.z));
    int3 surface_select = { // only one dim should take +1, -1 and others should take 0.
        coord.x / maximum, // +1 -> [X+], -1 -> [X-]
        coord.y / maximum, // +1 -> [Y+], -1 -> [Y-]
        coord.z / maximum  // +1 -> [Z+], -1 -> [Z-]
    }

    if (surface_select.x == 1 || surface_select.x == -1) // [X+] or [X-]
    {

    }


}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    const dim3 block(16,16); // blockDim.x, blockDim.y
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    cubetoequi<<<grid, block>>>(src, dst, src.rows, src.cols, src.cols/3);
}