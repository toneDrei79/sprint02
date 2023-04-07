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

__device__ float3 project_x(float theta, float phi, float rho)
{
    float3 coord = {
        rho,
        rho * sin(theta) * sin(phi),
        rho * cos(phi)
    };
    return coord;
}

__device__ float3 project_y(float theta, float phi, float rho)
{
    float3 coord = {
        rho * cos(theta) * sin(phi),
        rho,
        rho * cos(phi)
    };
    return coord;
}

__device__ float3 project_z(float theta, float phi, float rho)
{
    float3 coord = {
        rho * cos(theta) * sin(phi),
        rho * sin(theta) * sin(phi),
        rho
    };
    return coord;
}

__global__ void cubetoequi(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int sqr)
{   
    // global dst coordinates
    const int2 g_coord = {
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y
    };

    // uv coordinates
    float2 uv = {
        float(g_coord.x) / (4*sqr),
        float(g_coord.y) / (2*sqr),
    };
    // printf("%f, %f\n", uv.x, uv.y);

    // polar angles
    float theta = uv.x * 2*PI;
    float phi = (1.-uv.y) * PI;
    float rho;
    // printf("%f, %f\n", theta, phi);

    // polar coordinates
    float3 p_coord = {
        cos(theta) * sin(phi),
        sin(theta) * sin(phi),
        cos(phi)
    };
    // printf("%f, %f, %f\n", p_coord.x, p_coord.y, p_coord.z);

    float maximum = max(max(abs(p_coord.x), abs(p_coord.y)), abs(p_coord.z));
    int3 surface_select = { // only one dim should take +1, -1 and others should take 0.
        p_coord.x / maximum, // +1 -> [X+], -1 -> [X-]
        p_coord.y / maximum, // +1 -> [Y+], -1 -> [Y-]
        p_coord.z / maximum  // +1 -> [Z+], -1 -> [Z-]
    };
    // printf("%d, %d, %d\n", surface_select.x, surface_select.y, surface_select.z);

    float3 l_src_coord; // local src coordinates in each segment
    int2 g_src_coord; // global src coordinates
    if (surface_select.x == 1 || surface_select.x == -1) // [X+] or [X-]
    {
        rho = (surface_select.x*0.5) / (cos(theta) * sin(phi));
        l_src_coord = project_x(theta, phi, rho);
        if (surface_select.x == 1) // [X+] -> top middle
        {
            g_src_coord.x = (l_src_coord.y+0.5)*sqr + sqr;
            g_src_coord.y = (l_src_coord.z+0.5)*sqr;
        }
        else // [X-] -> bottom left
        {
            g_src_coord.x = (-l_src_coord.y+0.5)*sqr;
            g_src_coord.y = (l_src_coord.z+0.5)*sqr + sqr;
        }
    }
    else if (surface_select.y == 1 || surface_select.y == -1) // [Y+] or [Y-]
    {
        rho = (surface_select.y*0.5) / (sin(theta) * sin(phi));
        l_src_coord = project_y(theta, phi, rho);
        if (surface_select.y == 1) // [Y+] -> top right
        { 
            g_src_coord.x = (-l_src_coord.x+0.5)*sqr + 2*sqr;
            g_src_coord.y = (l_src_coord.z+0.5)*sqr;
        }
        else // [Y-] -> top left
        {
            g_src_coord.x = (l_src_coord.x+0.5)*sqr;
            g_src_coord.y = (l_src_coord.z+0.5)*sqr;
        }
    }
    else // [Z+] or [Z-]
    {
        rho = (surface_select.z*0.5) / (cos(phi));
        l_src_coord = project_z(theta, phi, rho);
        if (surface_select.z == 1) // [Z+] -> bottom right
        {
            g_src_coord.x = (l_src_coord.y+0.5)*sqr + sqr;
            g_src_coord.y = (-l_src_coord.x+0.5)*sqr + sqr;
        }
        else // [Z-] -> bottom middle
        {
            g_src_coord.x = (l_src_coord.y+0.5)*sqr + 2*sqr;
            g_src_coord.y = (l_src_coord.x+0.5)*sqr + sqr;
        }
    }

    // if (surface_select.x == 1)
    // if (surface_select.x == -1)
    // if (surface_select.y == 1)
    // if (surface_select.y == -1)
    // if (surface_select.z == 1)
    // if (surface_select.z == -1)
        // printf("[%d %d %d]: %f, %f -> %d, %d\n", surface_select.x, surface_select.y, surface_select.z, l_src_coord.x, l_src_coord.y, g_src_coord.x, g_src_coord.y);

    dst(g_coord.y, g_coord.x).x = src(g_src_coord.y, g_src_coord.x).x;
    dst(g_coord.y, g_coord.x).y = src(g_src_coord.y, g_src_coord.x).y;
    dst(g_coord.y, g_coord.x).z = src(g_src_coord.y, g_src_coord.x).z;
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