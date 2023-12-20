/*! \file multiGPU.h
 *  \brief kernels of for warming up and running on GPU
 */

#ifndef MULTIGPU_H
#define MULTIGPU_H

#include <gpuCommon.h>
#include <profile_util.h>
#include <string>
#include <map>
#include <thread>
#include <logger.h>

/// \defgroup kernels
/// GPU kernels 
//@{
/// @brief run the full vector add set of instructions (allocation, mem copies, kernel) running on all devices, logging information
/// @param logger Logger instance for logging info
/// @param Niter number of iterations instance for logging info
/// @param N size of vectors
void run_on_devices(Logger &logger, int Niter, int N);
/// @brief run the full vector add set of instructions (allocation, mem copies, kernel) returning statistics
/// @param val offset value to add to the initial vectors before running the add
/// @param N size of vectors
/// @param gridSize thread grid size
/// @param blockSize block size per thread
/// @return map containing the statisitics of a given set of instructions 
std::map<std::string, double> run_kernel(int val, int N, int gridSize, int blockSize);
std::map<std::string, double> run_kernel_without_allocation(
    int val, int N, int gridSize, int blockSize,
    float *&x, float *&y, float *&out, float *&d_x, float *&d_y, float *&d_out
    );
std::map<std::string, double> run_kernel_without_transfer(
    int val, int N, int gridSize, int blockSize,
    float *&x, float *&y, float *&out, float *&d_x, float *&d_y, float *&d_out
    );
//@}

#endif
