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

#define serial_code
__global__
void computeMask(uchar4* d_sourceImage, unsigned char *d_mask, int numRows, int numCols)
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
void computeBoundary(unsigned char *d_mask, unsigned char *d_borderPixels,
                     unsigned char *d_interiorPixels,
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


__global__
void initBufs(const uchar4 * const d_sourceImg, float *d_RedBuf,
              float *d_GreenBuf, float *d_BlueBuf, size_t srcSize)
{
    int tId = blockIdx.x * blockDim.x + threadIdx.x;

    if (tId > srcSize)
        return;

  uchar4 p = d_sourceImg[tId];
  d_RedBuf[tId] = p.x;
  d_GreenBuf[tId] = p.y;
  d_BlueBuf[tId] = p.z;

}

__global__
void computeIteration(unsigned char* d_interiorPixels, unsigned char *d_destImg,
                      float *d_buf1, float *d_buf2, float *g,
                      int numRows, int numCols)
{
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId >= numRows*numCols || (!d_interiorPixels[tId]))
        return;
    float blended_sum = 0.f;
    float border_sum = 0.f;

    if (d_interiorPixels[tId])
    {
        if (d_interiorPixels[tId + 1])
            blended_sum += d_buf1[tId + 1];
        else
            border_sum += d_destImg[tId + 1];

        if (d_interiorPixels[tId - 1])
            blended_sum += d_buf1[tId - 1];
        else
            border_sum += d_destImg[tId - 1];

        if (d_interiorPixels[tId - numCols])
            blended_sum += d_buf1[tId - numCols];
        else
            border_sum += d_destImg[tId - numCols];

        if (d_interiorPixels[tId + numCols])
            blended_sum += d_buf1[tId + numCols];
        else
            border_sum += d_destImg[tId + numCols];
    }
    float f_next_val = (blended_sum + border_sum + g[tId]) / 4.f;
    d_buf2[tId] = min(255.f, max(0.f, f_next_val)); //clip to

}

__global__
void computeG(unsigned char *d_channel, float *d_G, unsigned char*d_interiorPixel,
              int numRows, int numCols)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if ((offset >= numRows * numCols) || (!d_interiorPixel[offset]))
        return;

    float sum = 4.f * d_channel[offset];

    sum -= (float)d_channel[offset - 1] + (float)d_channel[offset + 1];
    sum -= (float)d_channel[offset + numCols] + (float)d_channel[offset - numCols];

    d_G[offset] = sum;
}

__global__
void swapBufs(float *d_buf1, float *d_buf2, int size)
{
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId >= size)
        return;

    float t = d_buf1[tId];
    d_buf1[tId] = d_buf2[tId];
    d_buf2[tId] = t;
}

__global__
void blendImages(uchar4 *d_destImg, unsigned char *d_interiorPixels,
                 float *d_RedBuf, float *d_GreenBuf, float *d_BlueBuf, int srcSize)
{
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if ((tId >= srcSize) || (!d_interiorPixels[tId]))
        return;

    d_destImg[tId].x = d_RedBuf[tId];
    d_destImg[tId].y = d_GreenBuf[tId];
    d_destImg[tId].z = d_BlueBuf[tId];

}

#ifdef serial_code
void compare_mask(unsigned char *source, const uchar4 *sourceImg, int srcSize)
{
  unsigned char *dest = new unsigned char[srcSize];
  for (int i = 0; i < srcSize; ++i) {
    dest[i] = (sourceImg[i].x + sourceImg[i].y + sourceImg[i].z < 3 * 255) ? 1 : 0;
  }

  for (int i = 0; i < srcSize; i++)
  {
    if (source[i] != dest[i])
      printf("Not matching \n");
  }
}

void compare_interior_border(unsigned char *cmp_interior, unsigned char *cmp_border, const uchar4 *sourceImg,
                             int numColsSource, int numRowsSource)
{
  unsigned char *mask = new unsigned char[numColsSource * numRowsSource];
  int srcSize = numColsSource * numRowsSource;
  for (int i = 0; i < srcSize; ++i) {
    mask[i] = (sourceImg[i].x + sourceImg[i].y + sourceImg[i].z < 3 * 255) ? 1 : 0;
  }
  unsigned char *borderPixels = new unsigned char[srcSize];
  unsigned char *strictInteriorPixels = new unsigned char[srcSize];

  std::vector<uint2> interiorPixelList;

  for (size_t r = 1; r < numRowsSource - 1; ++r) {
    for (size_t c = 1; c < numColsSource - 1; ++c) {
      if (mask[r * numColsSource + c]) {
        if (mask[(r -1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
            mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1]) {
          strictInteriorPixels[r * numColsSource + c] = 1;
          borderPixels[r * numColsSource + c] = 0;
          interiorPixelList.push_back(make_uint2(r, c));
        }
        else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 1;
        }
      }
      else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 0;

      }
    }
 }

    for (int i = 0; i < numRowsSource * numColsSource; i++)
    {
      if (strictInteriorPixels[i] != cmp_interior[i])
          printf("Inter unmatched \n");
      if (borderPixels[i] != cmp_border[i])
          printf("Border unmatched \n");
    }

}

void compare_RGB(unsigned char *cmp_R, unsigned char *cmp_G, unsigned char*cmp_B,
                 const uchar4 *sourceImg, int srcSize)
{
  unsigned char* red_src   = new unsigned char[srcSize];
  unsigned char* blue_src  = new unsigned char[srcSize];
  unsigned char* green_src = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    red_src[i]   = sourceImg[i].x;
    green_src[i]  = sourceImg[i].y;
    blue_src[i] = sourceImg[i].z;
  }

  for (int i = 0; i < srcSize; ++i) {
    if (red_src[i] != cmp_R[i])
      printf("Red unmatched \n");
    if (green_src[i] != cmp_G[i])
      printf("Green unmatched \n");
    if (blue_src[i] != cmp_B[i])
      printf("Blue unmatched \n");
  }
  unsigned char* red_dst   = new unsigned char[srcSize];
  unsigned char* blue_dst  = new unsigned char[srcSize];
  unsigned char* green_dst = new unsigned char[srcSize];
/*
  for (int i = 0; i < srcSize; ++i) {
    red_dst[i]   = destImg[i].x;
    blue_dst[i]  = destImg[i].y;
    green_dst[i] = destImg[i].z;
  }*/

}

void computeG_serial(const unsigned char* const channel,
              float* const g,
              const size_t numColsSource,
              const std::vector<uint2>& interiorPixelList)
{
  for (size_t i = 0; i < interiorPixelList.size(); ++i) {
    uint2 coord = interiorPixelList[i];
    unsigned int offset = coord.x * numColsSource + coord.y;

    float sum = 4.f * channel[offset];

    sum -= (float)channel[offset - 1] + (float)channel[offset + 1];
    sum -= (float)channel[offset + numColsSource] + (float)channel[offset - numColsSource];

    g[offset] = sum;
  }
}

void compare_jacobi(float *cmp_buf1, float *cmp_buf2, const uchar4 *h_sourceImg,
                    const uchar4 *h_destImg, int numColsSource, int numRowsSource)
{

  size_t srcSize = numRowsSource * numColsSource;
  unsigned char* mask = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    mask[i] = (h_sourceImg[i].x + h_sourceImg[i].y + h_sourceImg[i].z < 3 * 255) ? 1 : 0;
  }

  unsigned char *borderPixels = new unsigned char[srcSize];
  unsigned char *strictInteriorPixels = new unsigned char[srcSize];

  std::vector<uint2> interiorPixelList;

  for (size_t r = 1; r < numRowsSource - 1; ++r) {
    for (size_t c = 1; c < numColsSource - 1; ++c) {
      if (mask[r * numColsSource + c]) {
        if (mask[(r -1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
            mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1]) {
          strictInteriorPixels[r * numColsSource + c] = 1;
          borderPixels[r * numColsSource + c] = 0;
          interiorPixelList.push_back(make_uint2(r, c));
        }
        else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 1;
        }
      }
      else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 0;

      }
    }
  }

  unsigned char* red_src   = new unsigned char[srcSize];
  unsigned char* blue_src  = new unsigned char[srcSize];
  unsigned char* green_src = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    red_src[i]   = h_sourceImg[i].x;
    green_src[i]  = h_sourceImg[i].y;
    blue_src[i] = h_sourceImg[i].z;
  }

  unsigned char* red_dst   = new unsigned char[srcSize];
  unsigned char* blue_dst  = new unsigned char[srcSize];
  unsigned char* green_dst = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    red_dst[i]   = h_destImg[i].x;
    green_dst[i]  = h_destImg[i].y;
    blue_dst[i] = h_destImg[i].z;
  }


  float *g_red   = new float[srcSize];
  float *g_blue  = new float[srcSize];
  float *g_green = new float[srcSize];

  memset(g_red,   0, srcSize * sizeof(float));
  memset(g_blue,  0, srcSize * sizeof(float));
  memset(g_green, 0, srcSize * sizeof(float));

  computeG_serial(red_src,   g_red,   numColsSource, interiorPixelList);
  computeG_serial(blue_src,  g_blue,  numColsSource, interiorPixelList);
  computeG_serial(green_src, g_green, numColsSource, interiorPixelList);

  /*for (int i = 0; i < srcSize; i++)
  {
    if (g_red[i] != cmp_gR[i])
      printf("Red unmatched \n");
    if (g_blue[i] != cmp_gB[i])
      printf("Blue unmatched \n");
    if (g_green[i] != cmp_gG[i])
      printf("Green unmatched \n");
  }*/
  float *blendedValsRed_1 = new float[srcSize];
  float *blendedValsRed_2 = new float[srcSize];

  float *blendedValsBlue_1 = new float[srcSize];
  float *blendedValsBlue_2 = new float[srcSize];

  float *blendedValsGreen_1 = new float[srcSize];
  float *blendedValsGreen_2 = new float[srcSize];

  //IC is the source image, copy over
  for (size_t i = 0; i < srcSize; ++i) {
    blendedValsRed_1[i] = red_src[i];
    blendedValsRed_2[i] = red_src[i];
    blendedValsBlue_1[i] = blue_src[i];
    blendedValsBlue_2[i] = blue_src[i];
    blendedValsGreen_1[i] = green_src[i];
    blendedValsGreen_2[i] = green_src[i];
  }
  const size_t numIterations = 800;
  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(red_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsRed_1, g_red,
                     blendedValsRed_2);

    std::swap(blendedValsRed_1, blendedValsRed_2);
  }
  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(green_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsGreen_1, g_green,
                     blendedValsGreen_2);

    std::swap(blendedValsGreen_1, blendedValsGreen_2);
  }
  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(blue_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsBlue_1, g_blue,
                     blendedValsBlue_2);

    std::swap(blendedValsBlue_1, blendedValsBlue_2);
  }


  for (int i = 0; i < srcSize; i++)
  {
      if (blendedValsBlue_1[i] != cmp_buf1[i])
          printf("Umatch1 \n");
      if (blendedValsBlue_2[i] != cmp_buf2[i])
          printf("Umatch2 \n");
  }


}
#endif

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  printf("%d %d\n", numRowsSource, numColsSource);


  uchar4 *d_sourceImg, *d_destImg;
  unsigned char* d_mask;
  unsigned char *d_RSrc, *d_GSrc, *d_BSrc;
  unsigned char *d_RDest, *d_GDest, *d_BDest;
  unsigned char *d_borderPixels, *d_interiorPixels;
  float *d_RedBuf1, *d_RedBuf2, *d_GreenBuf1, *d_GreenBuf2, *d_BlueBuf1, *d_BlueBuf2;
  float *d_gRed, *d_gGreen, *d_gBlue;

  size_t srcSize = numRowsSource * numColsSource;

  checkCudaErrors(cudaMalloc((void **)&d_sourceImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * srcSize,
                            cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void **)&d_destImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * srcSize,
                            cudaMemcpyHostToDevice));


  { //step 1: compute Mask
  checkCudaErrors(cudaMalloc((void **)&d_mask, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMemset(d_mask, 0, sizeof(unsigned char) * srcSize));

  dim3 blockSize(1024, 1, 1);
  dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);
  computeMask<<<gridSize, blockSize>>>(d_sourceImg, d_mask, numRowsSource, numColsSource);

#ifdef serial_code1
  unsigned char *cmp_mask = new unsigned char[srcSize];
  cudaMemcpy(cmp_mask, d_mask, sizeof(unsigned char) * srcSize, cudaMemcpyDeviceToHost);
  compare_mask(cmp_mask, h_sourceImg, srcSize);
#endif
  }

  { //step2: compute interior and border pixels
  checkCudaErrors(cudaMalloc((void **)&d_borderPixels, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMemset(d_borderPixels, 0, sizeof(unsigned char) * srcSize));

  checkCudaErrors(cudaMalloc((void **)&d_interiorPixels, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMemset(d_interiorPixels, 0, sizeof(unsigned char) * srcSize));

  dim3 blockSize(32, 32, 1);
  dim3 gridSize((numRowsSource + 32 - 1)/32, (numColsSource + 32 - 1)/32, 1);
  computeBoundary<<<gridSize, blockSize>>>(d_mask, d_borderPixels, d_interiorPixels,
                                           numRowsSource, numColsSource);
  checkCudaErrors(cudaGetLastError());
#ifdef serial_code1
  unsigned char *cmp_interior = new unsigned char[srcSize];
  unsigned char *cmp_border = new unsigned char[srcSize];
  cudaMemcpy(cmp_interior, d_interiorPixels, sizeof(unsigned char) * srcSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(cmp_border, d_borderPixels, sizeof(unsigned char) * srcSize, cudaMemcpyDeviceToHost);
  compare_interior_border(cmp_interior, cmp_border, h_sourceImg, numColsSource, numRowsSource);
#endif
  }

  { //step 3: separate channels
  checkCudaErrors(cudaMalloc((void **)&d_RSrc, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_GSrc, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_BSrc, sizeof(unsigned char) * srcSize));

  dim3 blockSize(1024, 1, 1);
  dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);

  seperateChannels<<<gridSize, blockSize>>>(d_sourceImg, d_RSrc, d_GSrc, d_BSrc, srcSize);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMalloc((void **)&d_RDest, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_GDest, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_BDest, sizeof(unsigned char) * srcSize));

  seperateChannels<<<gridSize, blockSize>>>(d_destImg, d_RDest, d_GDest, d_BDest, srcSize);
  checkCudaErrors(cudaGetLastError());
#ifdef serial_code1
  unsigned char *cmp_R = new unsigned char[srcSize];
  unsigned char *cmp_G = new unsigned char[srcSize];
  unsigned char *cmp_B = new unsigned char[srcSize];
  cudaMemcpy(cmp_R, d_RSrc, sizeof(unsigned char) * srcSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(cmp_G, d_GSrc, sizeof(unsigned char) * srcSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(cmp_B, d_BSrc, sizeof(unsigned char) * srcSize, cudaMemcpyDeviceToHost);
  compare_RGB(cmp_R, cmp_G, cmp_B, h_sourceImg, srcSize);
#endif
  }

  {
    //step 3':
  checkCudaErrors(cudaMalloc((void **)&d_gRed, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_gGreen, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc((void **)&d_gBlue, sizeof(float) * srcSize));

  cudaMemset(d_gRed, 0, sizeof(float) * srcSize);
  cudaMemset(d_gGreen, 0, sizeof(float) * srcSize);
  cudaMemset(d_gBlue, 0, sizeof(float) * srcSize);

  dim3 blockSize(1024, 1, 1);
  dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);
  computeG<<<gridSize, blockSize>>>(d_RSrc, d_gRed, d_interiorPixels,
                                    numRowsSource, numColsSource);
  computeG<<<gridSize, blockSize>>>(d_GSrc, d_gGreen, d_interiorPixels,
                                    numRowsSource, numColsSource);
  computeG<<<gridSize, blockSize>>>(d_BSrc, d_gBlue, d_interiorPixels,
                                    numRowsSource, numColsSource);

  }

  { //step 4: allocate buffers
    checkCudaErrors(cudaMalloc((void **)&d_RedBuf1, sizeof(float)*srcSize));
    checkCudaErrors(cudaMalloc((void **)&d_RedBuf2, sizeof(float)*srcSize));

    checkCudaErrors(cudaMalloc((void **)&d_GreenBuf1, sizeof(float)*srcSize));
    checkCudaErrors(cudaMalloc((void **)&d_GreenBuf2, sizeof(float)*srcSize));

    checkCudaErrors(cudaMalloc((void **)&d_BlueBuf1, sizeof(float)*srcSize));
    checkCudaErrors(cudaMalloc((void **)&d_BlueBuf2, sizeof(float)*srcSize));


    dim3 blockSize(1024, 1, 1);
    dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);
    initBufs<<<gridSize, blockSize>>>(d_sourceImg, d_RedBuf1, d_GreenBuf1, d_BlueBuf1, srcSize);
    initBufs<<<gridSize, blockSize>>>(d_sourceImg, d_RedBuf2, d_GreenBuf2, d_BlueBuf2, srcSize);
    checkCudaErrors(cudaGetLastError());
    }

    {//step 5: Jacobi iterations
        dim3 blockSize(1024, 1, 1);
        dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);

        for (unsigned int i = 0; i < 800; i++)
        {
            computeIteration<<<gridSize, blockSize>>>(d_interiorPixels, d_RDest,
                                                      d_RedBuf1, d_RedBuf2, d_gRed,
                                                      numRowsSource, numColsSource);
            swapBufs<<<gridSize, blockSize>>>(d_RedBuf1, d_RedBuf2, srcSize);
        }

        for (unsigned int i = 0; i < 800; i++)
        {
            computeIteration<<<gridSize, blockSize>>>(d_interiorPixels, d_GDest,
                                                      d_GreenBuf1, d_GreenBuf2, d_gGreen,
                                                      numRowsSource, numColsSource);
            swapBufs<<<gridSize, blockSize>>>(d_GreenBuf1, d_GreenBuf2, srcSize);
        }

        for (unsigned int i = 0; i < 800; i++)
        {
            computeIteration<<<gridSize, blockSize>>>(d_interiorPixels, d_BDest,
                                                      d_BlueBuf1, d_BlueBuf2, d_gBlue,
                                                      numRowsSource, numColsSource);
            swapBufs<<<gridSize, blockSize>>>(d_BlueBuf1, d_BlueBuf2, srcSize);
        }
#ifdef serial_code1
    float *cmp_buf1 = new float[srcSize];
    float *cmp_buf2 = new float[srcSize];
    cudaMemcpy(cmp_buf1, d_BlueBuf1, sizeof(float) * srcSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(cmp_buf2, d_BlueBuf2, sizeof(float) * srcSize, cudaMemcpyDeviceToHost);
    compare_jacobi(cmp_buf1, cmp_buf2, h_sourceImg,
                  h_destImg, numColsSource, numRowsSource);

#endif
    }

    {
        //step 6: replace all interiors with above computed values.
        dim3 blockSize(1024, 1, 1);
        dim3 gridSize((numRowsSource*numColsSource + 1024-1)/1024, 1, 1);
        blendImages<<<gridSize, blockSize>>>(d_destImg, d_interiorPixels, d_RedBuf1,
                                             d_GreenBuf1, d_BlueBuf1, srcSize);

        cudaMemcpy(h_blendedImg, d_destImg, sizeof(uchar4) * srcSize,
                   cudaMemcpyDeviceToHost);

    }


  cudaFree(d_sourceImg);
  cudaFree(d_destImg);
  cudaFree(d_mask);
  cudaFree(d_borderPixels);
  cudaFree(d_interiorPixels);
  cudaFree(d_RSrc);
  cudaFree(d_GSrc);
  cudaFree(d_BSrc);
  cudaFree(d_RDest);
  cudaFree(d_GDest);
  cudaFree(d_BDest);
  cudaFree(d_RedBuf1);
  cudaFree(d_RedBuf2);
  cudaFree(d_GreenBuf1);
  cudaFree(d_GreenBuf2);
  cudaFree(d_BlueBuf1);
  cudaFree(d_BlueBuf2);
  cudaFree(d_gRed);
  cudaFree(d_gBlue);
  cudaFree(d_gGreen);


  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();

}

