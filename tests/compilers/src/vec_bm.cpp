#include <vector>
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <random>
#include "profile_util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MPI
#include <mpi.h>
#endif

// save typing
using vec_type = std::vector<float>;
using timer_type = profiling_util::Timer; 

// Allocate memory for the vectors
void allocateMem(vec_type& a, vec_type& b, vec_type& c, vec_type& d, unsigned int Nentries)
{
    // Start timer
    timer_type t(__func__, std::to_string(__LINE__));

    // Allocate memory
    a.resize(Nentries);
    b.resize(Nentries);
    c.resize(Nentries);
    d.resize(Nentries);

    // Report time taken to allocate memory
    std::string time_report = profiling_util::ReportTimeTaken(t, __func__, std::to_string(__LINE__));
    std::cout << time_report << std::endl;
}

// Initialise the vectors with randomly generated data
void initialiseVecs(vec_type& a, vec_type& b, vec_type& c, unsigned int Nentries)
{
    // Start timer
    timer_type t(__func__, std::to_string(__LINE__));

    // Use OMP parallelisation for actual generation of data
#ifdef _OMP
#pragma omp parallel
{
#endif
    // Generate random numbers for the vector entries
    srand(time(NULL));
    auto seed = rand(); // Seed random number generator
    std::default_random_engine generator(seed);
    // Draw numbers from a unfiorm distribution U(umin, umax)
    std::uniform_real_distribution<float> udist(0, 100);

#ifdef _OMP
    #pragma omp for
#endif
    // Populate the vectors
    for (auto i = 0; i < Nentries; i++)
    {
        a[i] = udist(generator);
        b[i] = udist(generator);
        c[i] = udist(generator);
    }
#ifdef _OMP
}
#endif

    // Report time taken to fill vectors
    std::string time_report = profiling_util::ReportTimeTaken(t, __func__, std::to_string(__LINE__));
    std::cout << time_report << std::endl;
}

// Perform simple math operations on the vectors
void doVecMath(vec_type& a, vec_type& b, vec_type& c, vec_type& d, unsigned int Nentries)
{
    // Start timer
    timer_type t(__func__, std::to_string(__LINE__));

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (auto i = 0; i < Nentries; i++) {
        d[i] = (a[i] + b[i]) * (c[i] * c[i] + b[i]);
    }

    // Time the vector math operation
    std::string time_report = profiling_util::ReportTimeTaken(t, __func__, std::to_string(__LINE__));
    std::cout << time_report << std::endl;
}


int main(int argc, char* argv[])
{

#ifdef _MPI
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    // Declare vectors
    std::vector<float> a, b, c, d;
    unsigned int Nentries = 100000;
    
    // Allocate memory
    allocateMem(a, b, c, d, Nentries);

    // Populate vectors
    initialiseVecs(a, b, c, Nentries);

    // Perform operations on vectors
    doVecMath(a, b, c, d, Nentries);

    // clear memory
    a.clear();
    b.clear();
    c.clear();
    d.clear();

#ifdef _MPI
    MPI_Finalize();
#endif

    // Message for ReFrame to search for in sanity tests
    std::cout << "Vector calculation finished" << std::endl;

    return 0;
}
