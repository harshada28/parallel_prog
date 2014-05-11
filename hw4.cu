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
void getPerElePerBlockHisto(unsigned int * const d_inputVals, int *d_histo, int *d_gCnt,
                            int numElem, int shift)
{
    
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask, val, index;
    __shared__ int cnt0, cnt1, cnt2, cnt3; //operating on 2-bit. Hence 4 combinations
    
    if (tId >= numElem)
        return;
    
    mask = 3 << shift;
    val = mask & d_inputVals[tId];
    switch (val)
    {
        case 0:
            atomicAdd(&cnt0, 1);
            atomicAdd(&(d_gCnt[0]), 1);
        break;
        
        case 1:
            atomicAdd(&cnt1, 1);
            atomicAdd(&(d_gCnt[1]), 1);
        break;
        
        case 2:
            atomicAdd(&cnt2, 1);
            atomicAdd(&(d_gCnt[2]), 1);
        break;
        
        case 3:
            atomicAdd(&cnt3, 1);
            atomicAdd(&(d_gCnt[3]), 1);
        
    }
    
    __syncthreads();
    if (tId == 0)
    {
        index = 0 * blockDim.x + blockIdx.x;
        d_histo[index] = cnt0;
        
        index = 1 * blockDim.x + blockIdx.x;
        d_histo[index] = cnt1;
       
        index = 2 * blockDim.x + blockIdx.x;
        d_histo[index] = cnt2;
       
        index = 3 * blockDim.x + blockIdx.x;
        d_histo[index] = cnt3;       
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
     int *d_globalHisto, *d_globalBins, *d_globalPrefixSum;
     int *h_binsPrefixSum;
     
     dim3 blockSize(1024, 1, 1);
     printf("Elements: %d BlockDim: %d NumBlocks: %d\n", numElems, blockSize.x,
                                      (numElems + blockSize.x - 1)/blockSize.x);
    
     checkCudaErrors(cudaMalloc((void **)&d_globalHisto, sizeof(int) * numBins * blockSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_globalBins, sizeof(int) * numBins));
     checkCudaErrors(cudaMalloc((void **)&d_globalPrefixSum, sizeof(int) * numBins * blockSize.x ));
    
     h_binsPrefixSum = (int *)malloc(sizeof(int) * numBins);
    
     for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i = i + numBits)
     {
         checkCudaErrors(cudaMemset(d_globalHisto, 0,  sizeof(int) * numBins * blockSize.x));
         checkCudaErrors(cudaMemset(d_globalPrefixSum, 0,  sizeof(int) * numBins * blockSize.x));
         checkCudaErrors(cudaMemset(d_globalBins, 0,  sizeof(int) * numBins));
         
         getPerElePerBlockHisto<<<(numElems + blockSize.x - 1)/blockSize.x, blockSize>>>
                                  (d_inputVals, d_globalHisto, d_globalBins, numBins, i);
         performParallelPrefixSum<<<numBins, (numElems + blockSize.x - 1)/blockSize.x,
                                   2 * sizeof(int) * (numElems + blockSize.x - 1)/blockSize.x>>>
                                  (d_globalHisto, d_globalPrefixSum);
         
         
         memset(h_binsPrefixSum, 0, sizeof(int) * numBins);
         checkCudaErrors(cudaMemcpy(h_binsPrefixSum, d_globalBins, sizeof(int) * numBins,
                                    cudaMemcpyDeviceToHost));
         performSerialPrefixSum(h_binsPrefixSum, numBins);
         //scatterElements<<<>>>     
     }
     checkCudaErrors(cudaFree(d_globalBins));
     checkCudaErrors(cudaFree(d_globalHisto));
     //checkCudaErrors(cudaFree(d_globalHisto));
     free(h_binsPrefixSum);
}

