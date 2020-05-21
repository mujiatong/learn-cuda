#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
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
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

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
λ nvprof add_cuda2.exe
==19500== NVPROF is profiling process 19500, command: add_cuda2.exe
Max error: 0
==19500== Profiling application: add_cuda2.exe
==19500== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  131.62us         1  131.62us  131.62us  131.62us  add(int, float*, float*)
      API calls:   75.64%  222.60ms         2  111.30ms  977.70us  221.62ms  cudaMallocManaged
                   18.73%  55.133ms         1  55.133ms  55.133ms  55.133ms  cuDevicePrimaryCtxRelease
                    4.51%  13.269ms         1  13.269ms  13.269ms  13.269ms  cudaLaunchKernel
                    0.85%  2.5044ms         2  1.2522ms  756.50us  1.7479ms  cudaFree
                    0.10%  286.50us        97  2.9530us     100ns  124.30us  cuDeviceGetAttribute
                    0.09%  256.50us         1  256.50us  256.50us  256.50us  cudaDeviceSynchronize
                    0.07%  199.50us         1  199.50us  199.50us  199.50us  cuModuleUnload
                    0.01%  23.600us         1  23.600us  23.600us  23.600us  cuDeviceTotalMem
                    0.00%  10.400us         1  10.400us  10.400us  10.400us  cuDeviceGetPCIBusId
                    0.00%  3.0000us         3  1.0000us     200ns  2.1000us  cuDeviceGetCount
                    0.00%  1.2000us         2     600ns     100ns  1.1000us  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid

==19500== Unified Memory profiling result:
Device "Quadro T1000 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     258  31.751KB  4.0000KB  32.000KB  8.000000MB  9.149400ms  Host To Device
     384  32.000KB  32.000KB  32.000KB  12.00000MB  78.10820ms  Device To Host
*/