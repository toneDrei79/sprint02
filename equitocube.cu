/*
    convert equiretangular iamge into cube map
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

__device__ float get_rho(float x, float y, float z)
{
    return sqrt(x*x + y*y + z*z);
}

__device__ float get_theta(float x, float y)
{
    if (y < 0)
    {
        return -atan2(y, x);
    }
    else
    {
        return PI + (PI - atan2(y, x));
    }
}

__device__ float get_phi(float z, float rho)
{
    return PI - acos(z/rho);
}

__global__ void equitocube(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int sqr)
{   
    /*
        +----+----+----+
        | Y+ | X+ | Y- |
        +----+----+----+
        | X- | Z- | Z+ |
        +----+----+----+

        consider the local coordinates in each segment
        these 6 segments can be considered as cubic space
    */

    // global dst coordinates
    const int2 g_coord = {
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y
    };
    
    // local dst coordinates
    int2 l_coord;

    // coordinates in cubic space
    int3 coord;

    if (g_coord.y < sqr + 1) // top
    {
        if (g_coord.x < sqr + 1) // left [Y-]
        {
            l_coord.x = g_coord.x;
            l_coord.y = g_coord.y;

            coord.x = l_coord.x - 0.5*sqr;
            coord.y = -0.5 * sqr;
            coord.z = l_coord.y - 0.5*sqr;
        }
        else if (g_coord.x < 2 * sqr + 1) // middle [X+]
        {
            l_coord.x = g_coord.x - sqr;
            l_coord.y = g_coord.y;
            
            coord.x = 0.5 * sqr;
            coord.y = l_coord.x - 0.5*sqr;
            coord.z = l_coord.y - 0.5*sqr;
        }
        else // right [Y+]
        {
            l_coord.x = g_coord.x - 2*sqr;
            l_coord.y = g_coord.y;

            coord.x = -(l_coord.x - 0.5*sqr);
            coord.y = 0.5 * sqr;
            coord.z = l_coord.y - 0.5*sqr;
        }
    }
    else // bottom
    {
        if (g_coord.x < sqr + 1) // left [X-]
        {
            l_coord.x = g_coord.x;
            l_coord.y = g_coord.y - sqr;

            coord.x = -0.5 * sqr;
            coord.y = -(l_coord.x - 0.5*sqr);
            coord.z = l_coord.y - 0.5*sqr;
        }
        else if (g_coord.x < 2 * sqr + 1) // middle [Z-]
        {
            l_coord.x = g_coord.x - sqr;
            l_coord.y = g_coord.y - sqr;

            coord.x = l_coord.y - 0.5*sqr;
            coord.y = l_coord.x - 0.5*sqr;
            coord.z = -0.5 * sqr;
        }
        else // right [Z+]
        {
            l_coord.x = g_coord.x - 2*sqr;
            l_coord.y = g_coord.y - sqr;

            coord.x = -(l_coord.y - 0.5*sqr);
            coord.y = l_coord.x - 0.5*sqr;
            coord.z = 0.5 * sqr;
        }
    }

    // polar angles
    float rho = get_rho(coord.x, coord.y, coord.z);
    float theta = get_theta(coord.x, coord.y) / (2*PI); // normalized angle 0.~1.
    float phi = get_phi(coord.z, rho) / PI; // normalized angle 0.~1.

    // printf("%f, %f, %f\n", rho, theta, phi);

    // reference coordinates
    int2 src_coord = {
        cols * theta,
        rows * phi 
    };

    // catch possible overflow
    if (src_coord.x >= cols)
    {
        src_coord.x -= cols;
    }
    if (src_coord.y >= rows)
    {
        src_coord.y -= rows;
    }

    dst(g_coord.y, g_coord.x).x = src(src_coord.y, src_coord.x).x;
    dst(g_coord.y, g_coord.x).y = src(src_coord.y, src_coord.x).y;
    dst(g_coord.y, g_coord.x).z = src(src_coord.y, src_coord.x).z;
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    const dim3 block(16,16); // blockDim.x, blockDim.y
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    equitocube<<<grid, block>>>(src, dst, src.rows, src.cols, dst.cols/3);
}