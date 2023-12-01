/*! \file WarmupGPU.h
 *  \brief kernels of for warming up and running on GPU
 */

#ifndef WARMUPGPU_H
#define WARMUPGPU_H

#include <gpuCommon.h>
#include <profile_util.h>
#include <string>
#include <map>
#include <logger.h>


/// GPU launch types
//@{
#define GPU_ONLY_KERNEL_LAUNCH 0 
#define GPU_ONLY_MEM_ALLOCATE 1
#define GPU_ONLY_MEM_TH2D 2
#define GPU_ONLY_MEM_TD2H 3
#define GPU_ONLY_NUM_LAUNCH_TYPES 4
//@}

/// \defgroup kernels
/// GPU kernels 
//@{

/// @brief launch a simple kernel or memory instruction to device
/// the instructions are 
/// a simple kernel, 
/// a on-device allocation and free
/// a transfer of host to device (th2d)
/// a transfer of host to device (td2h)
/// @param itype type of GPU instruction
/// @param device_id device id
/// @param round_id round of running the kernel  
/// @param N size of vectors
void launch_warmup_kernel(int itype, int device_id, int round_id, unsigned long long N);
/// @brief run several rounds of instructions, swaping kernels within a given round 
/// @param rounds number of rounds, default 2
/// @param N size of vectors used in allocation
void warmup_kernel_over_kernels(int rounds = 2, 
    std::vector<int> kernel_order = {GPU_ONLY_KERNEL_LAUNCH, GPU_ONLY_MEM_ALLOCATE, GPU_ONLY_MEM_TH2D, GPU_ONLY_MEM_TD2H}, 
    unsigned long long N = 1048576
);
/// @brief run several rounds of instructions doing rounds before swapping kernel type
/// @param rounds number of rounds
/// @param sleeptime whether to sleep between rounds 
void warmup_kernel_over_rounds(int rounds = 2, int sleeptime = 0, unsigned long long N = 1048576);
/// @brief run the full vector add set of instructions (allocation, mem copies, kernel) running on all devices, logging information 
/// @param Logger Logger instance for logging info
/// @param Niter number of iterations of the run kernel 
void run_on_devices(Logger &, int);
/// @brief run the full vector add set of instructions (allocation, mem copies, kernel) returning statistics
/// @param val offset value to add to the initial vectors before running the add
/// @return map containing the statisitics of a given set of instructions 
std::map<std::string, double> run_kernel(int val);

//void run_memcopy();
//@}

#endif
