/*! \file gpuCommon.h
 *  \brief common gpu related items
 */

#ifndef GPUCOMMON_H
#define GPUCOMMON_H

#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENACC
#include <openacc.h>
#endif

#ifdef USEHIP

#include <hip/hip_runtime.h>

#define gpuGetDeviceCount hipGetDeviceCount
#define gpuSetDevice hipSetDevice
#define gpuMemGetInfo hipMemGetInfo
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuHostAlloc(ptr, size) hipHostMalloc(ptr, size)
#define gpuHostFree hipHostFree
#define gpuFreeAsync hipHostFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize  hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#elif defined(USECUDA)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuSetDevice cudaSetDevice
#define gpuMemGetInfo cudaMemGetInfo
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuHostAlloc(ptr, size) cudaHostAlloc(ptr, size, cudaHostAllocDefault)
#define gpuHostFree cudaHostFree
#define gpuFreeAsync(ptr) cudaFreeAsync(ptr, 0)
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime

#endif

#define LogGPUElapsedTime(descrip,t1) {std::cout<<descrip<<" ::";LogTimeTakenOnDevice(t1);}

#endif