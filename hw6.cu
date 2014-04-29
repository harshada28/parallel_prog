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
void seperateChannels(char *d_sourceImg, char *d_RSrc, char *d_GSrc, char *d_BSrc)
{
  int tId = blockIdx.x * blockDim.x + threadIdx.x;
    
  if (tId > size)
    return;

  uchar4 p = d_sourceImage[tId];
  d_RSrc[tId] = p.x;
  d_GSrc[tId] = p.y;
  d_BSrc[tId] = p.z;
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


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

    printf("%d %d\n", numRowsSource, numColsSource);
    

  uchar4* d_sourceImg;
  size_t srcSize = numRowsSource * numColsSource;

  checkCudaErrors(cudaMalloc((void **)&d_sourceImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * srcSize,
                            cudaMemcpyHostToDevice));

  char* d_mask;
  checkCudaErrors(cudaMalloc((void **)&d_mask, sizeof(char) * srcSize));
  checkCudaErrors(cudaMemset(d_mask, 0, sizeof(char) * srcSize));

  dim3 blockSize(1024, 1, 1);
  dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);
  computeMask<<<gridSize, blockSize>>>(d_sourceImg, d_mask, numRowsSource, numColsSource);

#ifdef serial1
  unsigned char* h_mask = new unsigned char[srcSize];
  unsigned char* cmp_mask = new unsigned char[srcSize];
  computeMask_Serial(h_sourceImg, h_mask, numRowsSource, numColsSource);
  checkCudaErrors(cudaMemcpy(cmp_mask, d_mask, sizeof(char) * srcSize, cudaMemcpyDeviceToHost));
  for (int i = 0; i < srcSize; i++)
  {
    if (cmp_mask[i] != h_mask[i])
      printf("Not matching \n");
  }
#endif  

  char *d_RSrc, *d_GSrc, *d_BSrc;
  checkCudaErrors(cudaMalloc((void **)&d_RSrc, sizeof(char) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_GSrc, sizeof(char) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_BSrc, sizeof(char) * srcSize));
  
  seperateChannels<<gridSize, blockSize>>>(d_sourceImg, d_RSrc, d_GSrc, d_BSrc);

  cudaFree(d_sourceImg);
  cudaFree(d_mask);
  cudaFree(d_RSrc);
  cudaFree(d_GSrc);
  cudaFree(d_BSrc);
  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();

}

