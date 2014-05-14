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
void performParallelPrefixSum(unsigned int *d_input, int numElem)
{
    int tId = threadIdx.x;
    extern __shared__ int shdata[];
    int *sdataIn = (int *)shdata;
    int *sdataOut = (int *)(&shdata[blockDim.x]);
    int step;

    if (tId >= numElem)
        return;
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
    d_input[tId] = sdataIn[tId];
}


__global__
void getCounts(unsigned int * const d_inputVals, unsigned int * d_gPSum,
               unsigned int *d_cnt0s, unsigned int *d_cnt1s, unsigned int *d_cnt2s, unsigned int *d_cnt3s,
               unsigned int *d_presum0, unsigned int *d_presum1, unsigned int *d_presum2, unsigned int *d_presum3,
               int numElem, int numBins, int shift)
{

    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask, val, index;
    __shared__ int cnt0, cnt1, cnt2, cnt3; //operating on 2-bit. Hence 4 combinations

/*    extern __shared__ int s_data[];
    int *s_presum0 = (int *)s_data;
    int *s_presum1 = (int *)(&s_data[blockDim.x]);
    int *s_presum2 = (int *)(&s_data[2*blockDim.x]);
    int *s_presum3 = (int *)(&s_data[3*blockDim.x]);
*/
    if (tId >= numElem)
        return;

    if (tId == 0)
      cnt0 = cnt1 = cnt2 = cnt3 = 0;

  /*  s_presum0[threadIdx.x] = 0;
    s_presum1[threadIdx.x] = 0;
    s_presum2[threadIdx.x] = 0;
    s_presum3[threadIdx.x] = 0;
    */
    __syncthreads();
    mask = 3 << shift;
    val = mask & d_inputVals[tId] >> shift;
    switch (val)
    {
        case 0:
            atomicAdd(&cnt0, 1);
            d_presum0[tId] = 1;
        break;

        case 1:
            atomicAdd(&cnt1, 1);
            d_presum1[tId] = 1;
        break;

        case 2:
            atomicAdd(&cnt2, 1);
            d_presum2[tId] = 1;
        break;

        case 3:
            atomicAdd(&cnt3, 1);
            d_presum3[tId] = 1;
    }
    __syncthreads();
    if (tId == 0)
    {
        atomicAdd(&d_gPSum[0], cnt0);
        atomicAdd(&d_gPSum[1], cnt1);
        atomicAdd(&d_gPSum[2], cnt2);
        atomicAdd(&d_gPSum[3], cnt3);

        d_cnt0s[blockIdx.x] = cnt0;
        d_cnt1s[blockIdx.x] = cnt1;
        d_cnt2s[blockIdx.x] = cnt2;
        d_cnt3s[blockIdx.x] = cnt3;
    }
}

__global__
void scatter(unsigned int *d_inputVals, unsigned int *d_inputPos, unsigned int *d_globalPrefixSum,
             unsigned int *d_blockwise0s, unsigned int *d_blockwise1s, unsigned int *d_blockwise2s, unsigned int *d_blockwise3s,
             unsigned int *d_lPrefixSum0, unsigned int *d_lPrefixSum1, unsigned int *d_lPrefixSum2, unsigned int *d_lPrefixSum3,
             unsigned int *d_ovals, unsigned int *d_opos,
             int numElem, int shift)
{
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int mask, val, new_index;

    if (tId >= numElem)
        return;
    mask = 3 << shift;
    val = (mask & d_inputVals[tId]) >> shift;
  //  printf("%d\n", val);
    #if 1
    switch (0)
    {
        case 0:
            new_index =  d_blockwise0s[blockIdx.x] + d_lPrefixSum0[threadIdx.x];
            //new_index =   0;//d_blockwise0s[0];
        break;

        case 1:
            new_index =  d_blockwise1s[blockIdx.x] + d_lPrefixSum1[threadIdx.x];
            //new_index =  0;//d_blockwise1s[0];
        break;

        case 2:
            new_index =  d_blockwise2s[blockIdx.x] + d_lPrefixSum2[threadIdx.x];
            //new_index =  0;//d_blockwise2s[0];
        break;

        case 3:
            //new_index =  d_blockwise3s[blockIdx.x] + d_lPrefixSum3[threadIdx.x];
            new_index = 0;//d_blockwise3s[0];
            break;
        default:
            printf("%d\n",val);
    }
    #endif
    new_index += d_globalPrefixSum[val];
    d_ovals[new_index] = d_inputVals[tId];
    d_opos[new_index] = d_inputPos[tId];

}

__global__
void mycopy(unsigned int *d_inputVals, unsigned int *d_inputPos,
            unsigned int *d_outputVals, unsigned int *d_outputPos, int numElem)
{
    int tId = blockIdx.x * blockDim.x + threadIdx.x;

    if (tId >= numElem)
        return;
    d_outputVals[tId] = d_inputVals[tId];
    d_outputPos[tId] = d_inputPos[tId];
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
     unsigned int *d_blockwise0s, *d_blockwise1s, *d_blockwise2s, *d_blockwise3s;
     unsigned int *d_lPrefixSum0, *d_lPrefixSum1, *d_lPrefixSum2, *d_lPrefixSum3;
     unsigned int *d_globalPrefixSum;
     unsigned int *h_globalPrefixSum;
     unsigned int *d_ipVals, *d_ipPos, *d_opVals, *d_opPos;

     /*unsigned int *h_inputVals = (unsigned int*)malloc(sizeof(unsigned int)*numElems);;
     cudaMemcpy(h_inputVals, d_inputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToHost);
     for (unsigned int z = 0; z < 8 * sizeof(unsigned int); z = z + 2)
     {

         int mask = 3 << z;
         for (int t = 0; t < numElems; t++)
         {

            int val = (mask & h_inputVals[t]) >> z;
            if (val > 3)
                printf("%d\n", val);
         }
     }
     return;*/
     dim3 blockSize(1024, 1, 1);
     dim3 gridSize((numElems + blockSize.x - 1)/blockSize.x, 1, 1);
     printf("Elements: %d BlockDim: %d NumBlocks: %d\n", numElems, blockSize.x,
                                      (numElems + blockSize.x - 1)/blockSize.x);

     checkCudaErrors(cudaMalloc((void **)&d_blockwise0s, sizeof(unsigned int) * gridSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_blockwise1s, sizeof(unsigned int) * gridSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_blockwise2s, sizeof(unsigned int) * gridSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_blockwise3s, sizeof(unsigned int) * gridSize.x ));
     checkCudaErrors(cudaMalloc((void **)&d_globalPrefixSum, sizeof(unsigned int) * numBins));
     checkCudaErrors(cudaMalloc((void **)&d_lPrefixSum0, sizeof(unsigned int) * numElems));
     checkCudaErrors(cudaMalloc((void **)&d_lPrefixSum1, sizeof(unsigned int) * numElems));
     checkCudaErrors(cudaMalloc((void **)&d_lPrefixSum2, sizeof(unsigned int) * numElems));
     checkCudaErrors(cudaMalloc((void **)&d_lPrefixSum3, sizeof(unsigned int) * numElems));

     checkCudaErrors(cudaMalloc((void **)&d_ipVals, sizeof(unsigned int) * numElems));
     checkCudaErrors(cudaMalloc((void **)&d_ipPos, sizeof(unsigned int) * numElems));
     checkCudaErrors(cudaMalloc((void **)&d_opVals, sizeof(unsigned int) * numElems));
     checkCudaErrors(cudaMalloc((void **)&d_opPos, sizeof(unsigned int) * numElems));

     h_globalPrefixSum = (unsigned int *)malloc(sizeof(unsigned int) * numBins);

     for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i = i + numBits)
     {
         checkCudaErrors(cudaMemset(d_globalPrefixSum, 0,  sizeof(unsigned int) * numBins));
         checkCudaErrors(cudaMemset(d_lPrefixSum0, 0,  sizeof(unsigned int) * numElems));
         checkCudaErrors(cudaMemset(d_lPrefixSum1, 0,  sizeof(unsigned int) * numElems));
         checkCudaErrors(cudaMemset(d_lPrefixSum2, 0,  sizeof(unsigned int) * numElems));
         checkCudaErrors(cudaMemset(d_lPrefixSum3, 0,  sizeof(unsigned int) * numElems));
         checkCudaErrors(cudaMemset(d_blockwise0s, 0,  sizeof(unsigned int) * gridSize.x));
         checkCudaErrors(cudaMemset(d_blockwise1s, 0,  sizeof(unsigned int) * gridSize.x));
         checkCudaErrors(cudaMemset(d_blockwise2s, 0,  sizeof(unsigned int) * gridSize.x));
         checkCudaErrors(cudaMemset(d_blockwise3s, 0,  sizeof(unsigned int) * gridSize.x));

         checkCudaErrors(cudaMemset(d_opVals, 0,  sizeof(unsigned int) * numElems));
         checkCudaErrors(cudaMemset(d_opPos, 0,  sizeof(unsigned int) * numElems));
         getCounts<<<gridSize, blockSize>>>(d_inputVals, d_globalPrefixSum, d_blockwise0s,
                                            d_blockwise1s, d_blockwise2s, d_blockwise3s,
                                            d_lPrefixSum0, d_lPrefixSum1, d_lPrefixSum2, d_lPrefixSum3,
                                            numElems, numBins, i);
         performParallelPrefixSum<<<gridSize, blockSize, 2 * sizeof(unsigned int) * blockSize.x>>>(d_lPrefixSum0, numElems);
         performParallelPrefixSum<<<gridSize, blockSize, 2 * sizeof(unsigned int) * blockSize.x>>>(d_lPrefixSum1, numElems);
         performParallelPrefixSum<<<gridSize, blockSize, 2 * sizeof(unsigned int) * blockSize.x>>>(d_lPrefixSum2, numElems);
         performParallelPrefixSum<<<gridSize, blockSize, 2 * sizeof(unsigned int) * blockSize.x>>>(d_lPrefixSum3, numElems);

         performParallelPrefixSum<<<1, gridSize, 2 * sizeof(unsigned int) * gridSize.x>>>
                                  (d_blockwise0s, 10000);
         performParallelPrefixSum<<<1, gridSize, 2 * sizeof(unsigned int) * gridSize.x>>>
                                  (d_blockwise1s, 10000);
         performParallelPrefixSum<<<1, gridSize, 2 * sizeof(unsigned int) * gridSize.x>>>
                                  (d_blockwise2s, 10000);
         performParallelPrefixSum<<<1, gridSize, 2 * sizeof(unsigned int) * gridSize.x>>>
                                  (d_blockwise3s, 10000);

         cudaMemcpy(h_globalPrefixSum, d_globalPrefixSum, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);
         unsigned int *h_temp = (unsigned int *)malloc(sizeof(unsigned int) * numBins);
         memset(h_temp, 0, sizeof(unsigned int) * numBins);
         for (int j = 1; j < numBins; j++)
            h_temp[j] = h_temp[j-1] + h_globalPrefixSum[j-1];

         cudaMemcpy(d_globalPrefixSum, h_temp, sizeof(unsigned int) * numBins, cudaMemcpyHostToDevice);
         mycopy<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, d_ipVals, d_ipPos, numElems);
         scatter<<<gridSize, blockSize>>>(d_ipVals, d_ipPos, d_globalPrefixSum, d_blockwise0s,
                                           d_blockwise1s, d_blockwise2s, d_blockwise3s,
                                          d_lPrefixSum0, d_lPrefixSum1, d_lPrefixSum2, d_lPrefixSum3,
                                          d_opVals, d_opPos,
                                          numElems, i);
        unsigned int *d_temp;
        d_temp = d_opVals;
        d_opVals = d_ipVals;
        d_ipVals = d_temp;

        d_temp = d_opPos;
        d_opPos = d_ipPos;
        d_ipPos = d_temp;

     }
     mycopy<<<gridSize, blockSize>>>(d_ipVals, d_ipPos, d_outputVals, d_outputPos, numElems);
     checkCudaErrors(cudaFree(d_globalPrefixSum));
     checkCudaErrors(cudaFree(d_lPrefixSum0));
     checkCudaErrors(cudaFree(d_lPrefixSum1));
     checkCudaErrors(cudaFree(d_lPrefixSum2));
     checkCudaErrors(cudaFree(d_lPrefixSum3));
     checkCudaErrors(cudaFree(d_blockwise0s));
     checkCudaErrors(cudaFree(d_blockwise1s));
     checkCudaErrors(cudaFree(d_blockwise2s));
     checkCudaErrors(cudaFree(d_blockwise3s));
     //checkCudaErrors(cudaFree(d_ipVals));
     //checkCudaErrors(cudaFree(d_ipPos));
     //free(h_binsPrefixSum);
}

