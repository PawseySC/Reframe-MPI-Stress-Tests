/*! \file main.cpp
    \brief run a test kernel on a GPU to see what the warm up period is 

*/


#include <multiGPU.h>
#include <logger.h>
#include <profile_util.h>

#include <string>
#include <iostream>
#include <complex>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <chrono>
#ifdef _OPENMP 
#include <omp.h>
#endif

/// @brief Run the program
/// @param argc number of args passed
/// @param argv pass the number of rounds, number of iterations of the full kernel and how the warm-up test should be run [0 is over rounds]
/// @return return error 
int main(int argc, char** argv)
{
    Logger logger;
    LogParallelAPI();
    LogBinding();
    auto runtype = logger.ReportGPUSetup();
    int Niter = 100;
    int size = 1024*1024;
    if (argc >= 2) Niter = atoi(argv[1]);
    if (argc >= 3) size = atoi(argv[2]);

    // run a kernel on all possible devices, report timings
    run_on_devices(logger, Niter, size);

    return 0;
}
