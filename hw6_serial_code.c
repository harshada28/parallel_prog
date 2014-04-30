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

