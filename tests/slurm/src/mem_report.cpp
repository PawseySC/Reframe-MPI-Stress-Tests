#include <vector>
#include <limits.h> // For INT_MIN and UINT_MAX
#include "profile_util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MPI
#include <mpi.h>
#endif


int main(int argc, char* argv[])
{
    std::string funcname = __func__; // main
    //cout << "Size of float (in bytes): " << sizeof(float) << '\n';

#ifdef _MPI
    int size, rank;
    char hostname[1024];
    int proc = sched_getcpu();
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    gethostname(hostname, 1024);
    printf("Rank %i of %i running on processor %i on %s.\n", rank, size, proc, hostname);
#endif

    // Convert command-line argument to int
    char* p;
    errno = 0; // not 'int errno', because the '#include' already defined it
    long arg = strtol(argv[1], &p, 10);
    if (*p != '\0' || errno != 0) {
        return 1; 
    }

    if (arg < INT_MIN || arg > UINT_MAX) {
        return 1;
    }
    unsigned int mem_footprint = arg;

    // Declare vectors
    // `mem_footprint` argument is in MB - need to convert to entries per vector
    std::vector<float> a, b, c, d;
    unsigned long long Nentries = mem_footprint * 1024.0 * 1024.0 / 4.0 / sizeof(float);

    // Allocate memory (and time it)
    std::cout << "--------------------------------------------------\n";
    std::cout << "Allocating " << mem_footprint << " MB of memory!\n";
    std::cout << "Requiring arrays of size " << Nentries << '\n';
    std::cout << "--------------------------------------------------\n";
    a.resize(Nentries);
    b.resize(Nentries);
    c.resize(Nentries);
    d.resize(Nentries);
    std::cout << "MPI " << rank << ": " << profiling_util::ReportMemUsage(funcname, std::to_string(__LINE__)) << '\n';

#ifdef _MPI
    MPI_Finalize();
#endif

    return 0;
}
