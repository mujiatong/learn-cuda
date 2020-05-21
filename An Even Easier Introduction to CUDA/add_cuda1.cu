#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 256>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}

/*
C:\tools
λ nvprof add_cuda1.exe
==29600== NVPROF is profiling process 29600, command: add_cuda1.exe
Max error: 0
==29600== Profiling application: add_cuda1.exe
==29600== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  2.1688ms         1  2.1688ms  2.1688ms  2.1688ms  add(int, float*, float*)
      API calls:   75.32%  217.13ms         2  108.57ms  968.50us  216.17ms  cudaMallocManaged
                   18.43%  53.141ms         1  53.141ms  53.141ms  53.141ms  cuDevicePrimaryCtxRelease
                    4.48%  12.926ms         1  12.926ms  12.926ms  12.926ms  cudaLaunchKernel
                    0.80%  2.3111ms         2  1.1555ms  710.70us  1.6004ms  cudaFree
                    0.79%  2.2862ms         1  2.2862ms  2.2862ms  2.2862ms  cudaDeviceSynchronize
                    0.10%  275.00us        97  2.8350us     100ns  120.80us  cuDeviceGetAttribute
                    0.05%  140.50us         1  140.50us  140.50us  140.50us  cuModuleUnload
                    0.01%  32.100us         1  32.100us  32.100us  32.100us  cuDeviceTotalMem
                    0.01%  14.800us         1  14.800us  14.800us  14.800us  cuDeviceGetPCIBusId
                    0.00%  3.5000us         2  1.7500us     500ns  3.0000us  cuDeviceGet
                    0.00%  1.5000us         3     500ns     200ns     700ns  cuDeviceGetCount
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid

==29600== Unified Memory profiling result:
Device "Quadro T1000 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     258  31.751KB  4.0000KB  48.000KB  8.000000MB  9.051700ms  Host To Device
     385  31.916KB  16.000KB  32.000KB  12.00000MB  77.45940ms  Device To Host
*/