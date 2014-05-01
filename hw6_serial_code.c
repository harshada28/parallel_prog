#ifdef serial
  unsigned char* h_mask = new unsigned char[srcSize];
  unsigned char* cmp_mask = new unsigned char[srcSize];
  computeMask_Serial(h_sourceImg, h_mask, numRowsSource, numColsSource);
  checkCudaErrors(cudaMemcpy(cmp_mask, d_mask, sizeof(char) * srcSize, cudaMemcpyDeviceToHost));
  /*for (int i = 0; i < srcSize; i++)
  {
    if (cmp_mask[i] != h_mask[i])
      printf("Not matching \n");
  }*/
#endif

#ifdef serial
  unsigned char *h_borderPixels = new unsigned char[srcSize];
  unsigned char *strictInteriorPixels = new unsigned char[srcSize];
  unsigned char *cmpPixels = new unsigned char[srcSize];

  checkCudaErrors(cudaMemcpy(cmpPixels, d_borderPixels, sizeof(unsigned char) * srcSize,
                             cudaMemcpyDeviceToHost));
   
  memset(h_borderPixels, 0, sizeof(unsigned char) * srcSize);  
  memset(strictInteriorPixels, 0, sizeof(unsigned char) * srcSize);  
  computeBorder_Serial(h_mask, h_borderPixels, strictInteriorPixels, numRowsSource, numColsSource);
  for (int i = 0; i < srcSize; i++)
  {
    //if (cmpPixels[i] != strictInteriorPixels[i])
      //  printf("Not matched \n");
    if (cmpPixels[i] != h_borderPixels[i])
        printf("Not matched \n");
      //printf("%d %d \n", cmpPixels[i], strictInteriorPixels[i]);
  }

  
#endif

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

