#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
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
  add<<<1, 1>>>(N, x, y);

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
λ nvprof add_cuda.exe
==12788== NVPROF is profiling process 12788, command: add_cuda.exe
Max error: 0
==12788== Profiling application: add_cuda.exe
==12788== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  123.79ms         1  123.79ms  123.79ms  123.79ms  add(int, float*, float*)
      API calls:   52.53%  214.84ms         2  107.42ms  2.5403ms  212.30ms  cudaMallocManaged
                   30.30%  123.94ms         1  123.94ms  123.94ms  123.94ms  cudaDeviceSynchronize
                   13.04%  53.347ms         1  53.347ms  53.347ms  53.347ms  cuDevicePrimaryCtxRelease
                    3.25%  13.287ms         1  13.287ms  13.287ms  13.287ms  cudaLaunchKernel
                    0.77%  3.1316ms         2  1.5658ms  805.40us  2.3262ms  cudaFree
                    0.06%  255.00us        97  2.6280us     100ns  109.50us  cuDeviceGetAttribute
                    0.04%  160.70us         1  160.70us  160.70us  160.70us  cuModuleUnload
                    0.01%  21.800us         1  21.800us  21.800us  21.800us  cuDeviceTotalMem
                    0.00%  10.600us         1  10.600us  10.600us  10.600us  cuDeviceGetPCIBusId
                    0.00%  1.2000us         2     600ns     200ns  1.0000us  cuDeviceGet
                    0.00%  1.1000us         3     366ns     200ns     500ns  cuDeviceGetCount
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

==12788== Unified Memory profiling result:
Device "Quadro T1000 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     258  31.751KB  4.0000KB  32.000KB  8.000000MB  9.132400ms  Host To Device
     384  32.000KB  32.000KB  32.000KB  12.00000MB  78.50020ms  Device To Host

*/
