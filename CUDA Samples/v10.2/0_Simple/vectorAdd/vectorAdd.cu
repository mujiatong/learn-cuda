/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 5000000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}




/*
¦Ë nvprof "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\bin\win64\Debug\vectorAdd.exe"
[Vector addition of 5000000 elements]
==13708== NVPROF is profiling process 13708, command: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\bin\win64\Debug\vectorAdd.exe
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 19532 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
==13708== Profiling application: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\bin\win64\Debug\vectorAdd.exe
==13708== Profiling result:
			Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.06%  13.006ms         1  13.006ms  13.006ms  13.006ms  vectorAdd(float const *, float const *, float*, int)
				   36.68%  12.534ms         2  6.2670ms  6.0185ms  6.5155ms  [CUDA memcpy HtoD]
				   25.26%  8.6321ms         1  8.6321ms  8.6321ms  8.6321ms  [CUDA memcpy DtoH]
	  API calls:   63.27%  190.04ms         3  63.348ms  2.0236ms  185.95ms  cudaMalloc
				   23.90%  71.783ms         1  71.783ms  71.783ms  71.783ms  cuDevicePrimaryCtxRelease
				   11.78%  35.381ms         3  11.794ms  6.2484ms  22.681ms  cudaMemcpy
					0.54%  1.6290ms         1  1.6290ms  1.6290ms  1.6290ms  cuModuleUnload
					0.32%  951.40us         3  317.13us  227.70us  407.30us  cudaFree
					0.14%  421.60us        97  4.3460us     100ns  140.80us  cuDeviceGetAttribute
					0.03%  81.900us         1  81.900us  81.900us  81.900us  cudaLaunchKernel
					0.01%  38.700us         1  38.700us  38.700us  38.700us  cuDeviceTotalMem
					0.00%  12.600us         1  12.600us  12.600us  12.600us  cuDeviceGetPCIBusId
					0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetLastError
					0.00%  2.3000us         3     766ns     400ns  1.4000us  cuDeviceGetCount
					0.00%  2.1000us         2  1.0500us     200ns  1.9000us  cuDeviceGet
					0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
					0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
					0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid

*/