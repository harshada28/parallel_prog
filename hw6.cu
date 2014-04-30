//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.

      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

#include <stdio.h>

#define serial
__global__
void computeMask(uchar4* d_sourceImage, char *d_mask, int numRows, int numCols)
{
  int tId = blockIdx.x * blockDim.x + threadIdx.x;
  if (tId > numRows*numCols)
    return;

  uchar4 p = d_sourceImage[tId];
  if (p.x + p.y + p.z < 3 * 255)
    d_mask[tId] = 1;
}

__global__
void seperateChannels(uchar4 *d_sourceImg, unsigned char *d_RSrc, unsigned char *d_GSrc,
                      unsigned char *d_BSrc,
                      size_t size)
{
  int tId = blockIdx.x * blockDim.x + threadIdx.x;

  if (tId > size)
    return;

  uchar4 p = d_sourceImg[tId];
  d_RSrc[tId] = p.x;
  d_GSrc[tId] = p.y;
  d_BSrc[tId] = p.z;
}

__global__
void computeBoundary(char *d_mask, char *d_borderPixels, char *d_interiorPixels,
                         int numRows, int numCols)
{
    const int2 coord =  make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y);
    if (coord.x >= numRows-1 || coord.y >= numCols-1
        || coord.x <= 0 || coord.y <=0)
        return;


    const int tId = coord.x * numCols + coord.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (d_mask[tId])
    {
        if (d_mask[(r -1) * numCols + c] && d_mask[(r + 1) * numCols + c] &&
            d_mask[r * numCols + c - 1] && d_mask[r * numCols + c + 1])
        {
            d_borderPixels[tId] = 0;
            d_interiorPixels[tId] = 1;
        }
        else
        {
            d_borderPixels[tId] = 1;
            d_interiorPixels[tId] = 0;
        }
    }

}


#ifdef serial
void computeMask_Serial(const uchar4 * const h_sourceImg, unsigned char* h_mask, int rows, int cols)
{
  size_t srcSize = rows * cols;

  for (int i = 0; i < srcSize; ++i) {
    h_mask[i] = (h_sourceImg[i].x + h_sourceImg[i].y + h_sourceImg[i].z < 3 * 255) ? 1 : 0;
  }

}
#endif

#ifdef serial
void computeBorder_Serial(unsigned char *mask, unsigned char *borderPixels,
                          unsigned char *interiorPixels,
                          int numRowsSource, int numColsSource)
{
    int sum = 0;
    for (size_t r = 1; r < numRowsSource - 1; ++r) {
    for (size_t c = 1; c < numColsSource - 1; ++c) {
      if (mask[r * numColsSource + c]) {
        if (mask[(r -1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
            mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1]) {
          interiorPixels[r * numColsSource + c] = 1;
          borderPixels[r * numColsSource + c] = 0;
            sum++;
          //interiorPixelList.push_back(make_uint2(r, c));
        }
        else {
          //strictInteriorPixels[r * numColsSource + c] = 0;
          interiorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 1;
        }
      }
      else {
          //strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 0;

      }
    }
  }

    printf("Host cnt: %d \n", sum);

}
#endif

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  printf("%d %d\n", numRowsSource, numColsSource);


  uchar4* d_sourceImg;
  char* d_mask;
  unsigned char *d_RSrc, *d_GSrc, *d_BSrc;
  char* d_borderPixels;
  char* d_interiorPixels;

  size_t srcSize = numRowsSource * numColsSource;

  checkCudaErrors(cudaMalloc((void **)&d_sourceImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * srcSize,
                            cudaMemcpyHostToDevice));

  { //step 1: compute Mask
  checkCudaErrors(cudaMalloc((void **)&d_mask, sizeof(char) * srcSize));
  checkCudaErrors(cudaMemset(d_mask, 0, sizeof(char) * srcSize));

  dim3 blockSize(1024, 1, 1);
  dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);
  computeMask<<<gridSize, blockSize>>>(d_sourceImg, d_mask, numRowsSource, numColsSource);
  }

  { //step2: compute interior and border pixels
  checkCudaErrors(cudaMalloc((void **)&d_borderPixels, sizeof(char) * srcSize));
  checkCudaErrors(cudaMemset(d_borderPixels, 0, sizeof(char) * srcSize));

  checkCudaErrors(cudaMalloc((void **)&d_interiorPixels, sizeof(char) * srcSize));
  checkCudaErrors(cudaMemset(d_interiorPixels, 0, sizeof(char) * srcSize));

  dim3 blockSize(32, 32, 1);
  dim3 gridSize((numRowsSource + 32 - 1)/32, (numColsSource + 32 - 1)/32, 1);
  computeBoundary<<<gridSize, blockSize>>>(d_mask, d_borderPixels, d_interiorPixels,
                                                 numRowsSource, numColsSource);
  checkCudaErrors(cudaGetLastError());
  }

  { //step 3: separate channels
  checkCudaErrors(cudaMalloc((void **)&d_RSrc, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_GSrc, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_BSrc, sizeof(unsigned char) * srcSize));

  dim3 blockSize(1024, 1, 1);
  dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);

  seperateChannels<<<gridSize, blockSize>>>(d_sourceImg, d_RSrc, d_GSrc, d_BSrc, srcSize);
  checkCudaErrors(cudaGetLastError());
  }


  cudaFree(d_sourceImg);
  cudaFree(d_mask);
  cudaFree(d_borderPixels);
  cudaFree(d_interiorPixels);
  cudaFree(d_RSrc);
  cudaFree(d_GSrc);
  cudaFree(d_BSrc);

  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();

}

