/*
    convert equiretangular iamge into cube map
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>

using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_src = cv::imread(argv[1]);

    float sqr = h_src.cols / 4.0;
    cv::Mat h_dst(int(sqr*2), int(sqr*3), CV_8UC3);

    cv::cuda::GpuMat d_src;
    cv::cuda::GpuMat d_dst;
    d_src.upload(h_src);
    d_dst.upload(h_dst);
        
    startCUDA(d_src, d_dst);
    d_dst.download(h_dst);
    
    cv::imshow("Original Image", h_src);
    cv::imshow("Processed Image", h_dst);
    cv::imwrite("images/image-cube.jpg", h_dst);

    cv::waitKey();

    return 0;
}