# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

<h3>ENTER YOUR NAME: RavivarmanVV</h3>
<h3>ENTER YOUR REGISTER NO: 212224240133</h3>
<h3>EX. NO: 3</h3>
<h3>DATE</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```python
%%writefile sobelEdgeDetectionFilter.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,
                            unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float Kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    float Ky[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {

        float Gx = 0.0f;
        float Gy = 0.0f;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {

                float pixel = srcImage[(y + ky) * width + (x + kx)];

                Gx += pixel * Kx[ky + 1][kx + 1];
                Gy += pixel * Ky[ky + 1][kx + 1];
            }
        }

        float G = sqrtf(Gx * Gx + Gy * Gy);

        if (G > 255.0f)
            G = 255.0f;

        dstImage[y * width + x] = (unsigned char)G;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {

    // Read input image
    Mat image = imread("/content/images.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory
    unsigned char *h_outputImage =
        (unsigned char *)malloc(imageSize);

    if (h_outputImage == NULL) {
        printf("Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;

    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));

    // Copy input image to GPU
    checkCudaErrors(cudaMemcpy(
        d_inputImage, image.data, imageSize,
        cudaMemcpyHostToDevice));

    // Initialize output image
    checkCudaErrors(cudaMemset(d_outputImage, 0, imageSize));

    // CUDA events for timing
    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Define block and grid size
    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + 15) / 16,
        (height + 15) / 16
    );

    // Launch CUDA kernel
    checkCudaErrors(cudaEventRecord(start));

    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage, d_outputImage, width, height);

    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Calculate execution time
    float milliseconds = 0.0f;

    checkCudaErrors(cudaEventElapsedTime(
        &milliseconds, start, stop));

    // Copy output from GPU to CPU
    checkCudaErrors(cudaMemcpy(
        h_outputImage, d_outputImage, imageSize,
        cudaMemcpyDeviceToHost));

    // Save output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);

    imwrite("/content/output_sobel.jpeg", outputImage);

    printf("Sobel Edge Detection Completed Successfully!\n");
    printf("Input Image: images.jpg\n");
    printf("Output Image: output_sobel.jpeg\n");
    printf("Total time taken: %f milliseconds\n", milliseconds);

    // Free memory
    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

```
## OUTPUT:
<img width="3840" height="2160" alt="images" src="https://github.com/user-attachments/assets/b5831a55-5069-4a3c-af29-20e1478f86ea" />
<img width="515" height="321" alt="image" src="https://github.com/user-attachments/assets/86115e5b-028e-4c75-b1e6-9d3868a894e3" />

## RESULT:
Thus the program has been executed by using CUDA to ________________.0.663776 milliseconds

Questions:

What challenges did you face while implementing the Sobel filter for color images?
How did changing the block size influence the performance of your CUDA implementation?
What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Suggest potential optimizations for improving the performance of the Sobel filter.

Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:

