#include <vector>
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MPI
#include <mpi.h>
#endif

// save typing
using vec_type = std::vector<float>;

// Initialise the vectors with randomly generated data
void initialiseVecs(vec_type& a, vec_type& b, vec_type& c, unsigned int Nentries)
{

#ifdef _OMP
#pragma omp parallel
{
#endif
    // Generate random numbers for the vector entries
    srand(time(NULL));
    auto seed = rand(); // Seed random number generator
    std::default_random_engine generator(seed);
    // Draw numbers from a uniform distribution U(umin, umax)
    std::uniform_real_distribution<float> udist(0, 100);

#ifdef _OMP
    #pragma omp for
#endif
    // Populate vectors
    for (auto i = 0; i < Nentries; i++)
    {
        a[i] = udist(generator);
        b[i] = udist(generator);
        c[i] = udist(generator);
    }
#ifdef _OMP
}
#endif

}

// Perform simple vector math operations on the vectors
void doVecMath(vec_type& a, vec_type& b, vec_type& c, vec_type& d, unsigned int Nentries)
{
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (auto i = 0; i < Nentries; i++) {
        d[i] = (a[i] + b[i]) * (c[i] * c[i] + b[i]);
    }

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
    a.resize(Nentries);
    b.resize(Nentries);
    c.resize(Nentries);
    d.resize(Nentries);

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

    // Message for ReFrame to search for in sanity test
    std::cout << "Vector calculation finished" << std::endl;

    return 0;
}
