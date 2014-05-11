//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

#include <stdio.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


__global__
void performParallelPrefixSum(int * const d_input, int * d_output)
{
    int tId = threadIdx.x;
    extern __shared__ int shdata[];
    int *sdataIn = (int *)shdata;
    int *sdataOut = (int *)(&shdata[blockDim.x]);
    int step;

    sdataIn[tId] = d_input[tId];
    sdataOut[tId] = sdataIn[tId];
    __syncthreads();
    for (unsigned int n = 0; ((pow(2.0, (double)n)) < blockDim.x); n++)
    {
        step = pow(2.0, (double)n);
        if (tId >= step)
        {
            sdataOut[tId] = sdataIn[tId] + sdataIn[tId-step];
        }
        __syncthreads();
        sdataIn[tId] = sdataOut[tId];
        __syncthreads();
    }
    d_output[tId] = sdataIn[tId];
}


__global__
void getCounts(unsigned int * const d_inputVals, int * d_gPSum,
               int *d_cnt0s, int *d_cnt1s, int *d_cnt2s, int *d_cnt3s,
               int numElem, int numBins, int shift)
{

    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask, val, index;
    __shared__ int cnt0, cnt1, cnt2, cnt3; //operating on 2-bit. Hence 4 combinations

    if (tId >= numElem)
        return;

    if (tId == 0)
      cnt0 = cnt1 = cnt2 = cnt3 = 0;

    __syncthreads();
    mask = 3 << shift;
    val = mask & d_inputVals[tId] >> shift;
    switch (val)
    {
        case 0:
            atomicAdd(&cnt0, 1);
        break;

        case 1:
            atomicAdd(&cnt1, 1);
        break;

        case 2:
            atomicAdd(&cnt2, 1);
        break;

        case 3:
            atomicAdd(&cnt3, 1);

    }

    __syncthreads();
    if (tId == 0)
    {
        atomicsAdd(&d_gPSum[0], cnt0);
        atomicsAdd(&d_gPSum[1], cnt1);
        atomicsAdd(&d_gPSum[2], cnt2);
        atomicsAdd(&d_gPSum[3], cnt3);

        d_cnt0s[blockIdx.x] = cnt0;
        d_cnt1s[blockIdx.x] = cnt1;
        d_cnt2s[blockIdx.x] = cnt2;
        d_cnt3s[blockIdx.x] = cnt3;
    }
}

void performSerialPrefixSum(int *h_globalPrefixSum, int numBins)
{
    int sum = 0;
    int *temp_buf = (int *)malloc(sizeof(int) * numBins);
    memset(temp_buf, 0, sizeof(int) * numBins);

    for (unsigned int i = 1; i < numBins; i++)
    {
        temp_buf[i] = sum;
        sum = sum + h_globalPrefixSum[i];
    }
    memcpy(h_globalPrefixSum, temp_buf, sizeof(int) * numBins);
    free(temp_buf);
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
     unsigned int numBits = 2;
     unsigned int numBins = 1 << numBits;
     int *d_blockwise0s, *d_blockwise1s, *d_blockwise2s, *d_blockwise3s,
     int *d_localPrefixSum, *d_globalPrefixSum;
     int *h_binsPrefixSum;

     dim3 blockSize(1024, 1, 1);
     dim3 gridSize((numElems + blockSize.x - 1)/blockSize.x, 1, 1);
     printf("Elements: %d BlockDim: %d NumBlocks: %d\n", numElems, blockSize.x,
                                      (numElems + blockSize.x - 1)/blockSize.x);

     checkCudaErrors(cudaMalloc((void **)&d_blockwise0s, sizeof(int) * gridSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_blockwise1s, sizeof(int) * gridSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_blockwise2s, sizeof(int) * gridSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_blockwise3s, sizeof(int) * gridSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_globalPrefixSum, sizeof(int) * numBins));
     checkCudaErrors(cudaMalloc((void **)&d_localPrefixSum, sizeof(int) * numElems));

     h_binsPrefixSum = (int *)malloc(sizeof(int) * numBins);

     for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i = i + numBits)
     {
         checkCudaErrors(cudaMemset(d_globalPrefixSum, 0,  sizeof(int) * numBins));
         checkCudaErrors(cudaMemset(d_localPrefixSum, 0,  sizeof(int) * numElems));
         checkCudaErrors(cudaMemset(d_blockwise0s, 0,  sizeof(int) * grdiSize.x));
         checkCudaErrors(cudaMemset(d_blockwise1s, 0,  sizeof(int) * grdiSize.x));
         checkCudaErrors(cudaMemset(d_blockwis2s, 0,  sizeof(int) * grdiSize.x));
         checkCudaErrors(cudaMemset(d_blockwise3s, 0,  sizeof(int) * grdiSize.x));

         getCounts<<<gridSize.x, blockSize>>>(d_inputVals, d_globalPrefixSum, d_blockwise0s,
                                              d_blockwise1s, d_blockwise2s, d_blockwise3s,
                                              numElems, numBins, i);
         performParallelPrefixSum<<<numBins, (numElems + blockSize.x - 1)/blockSize.x,
                                   2 * sizeof(int) * (numElems + blockSize.x - 1)/blockSize.x>>>
                                  (d_globalHisto, d_globalPrefixSum);

        #if 0
         memset(h_binsPrefixSum, 0, sizeof(int) * numBins);
         checkCudaErrors(cudaMemcpy(h_binsPrefixSum, d_globalBins, sizeof(int) * numBins,
                                    cudaMemcpyDeviceToHost));
         performSerialPrefixSum(h_binsPrefixSum, numBins);
        #endif
         //scatterElements<<<>>>
     }
     checkCudaErrors(cudaFree(d_globalPrefixSum));
     checkCudaErrors(cudaFree(d_localPrefixSum));
     checkCudaErrors(cudaFree(d_blockwise0s));
     checkCudaErrors(cudaFree(d_blockwise1s));
     checkCudaErrors(cudaFree(d_blockwise2s));
     checkCudaErrors(cudaFree(d_blockwise3s));
     free(h_binsPrefixSum);
}

