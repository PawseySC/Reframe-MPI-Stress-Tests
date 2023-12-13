#include "profile_util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MPI
#include <mpi.h>
#endif


int main(int argc, char* argv[])
{

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

    // Report Parallel API, affinity, and binding information
    std::string api_string = profiling_util::ReportParallelAPI();
    std::cout << api_string << std::endl;

#ifdef _OPENMP
  #pragma omp parallel
  {
    // Report thread affinity (must be done inside parallel region)
    // Report within critical region so multiple threads don't try
    // to report at once, screwing up the formatting of the output
    #pragma omp critical
    {
    std::string thread_affinity = profiling_util::ReportThreadAffinity(__func__, 
                                  std::to_string(__LINE__));
    // When ranks run in parallel, their output can be interleaved amongst each other
    // Adding the MPI rank to the output allows one to know which rank is "talking"
    // when.
    // TODO: Add this reporting to `profile_util` library
#ifdef _MPI
    std::cout << "Node " << hostname << ": MPI " << rank << ": " << thread_affinity << std::endl;
#else
    std::cout << thread_affinity << std::endl;
#endif
    }
  }
  // Report binding (must be done outside parallel region)
  std::string binding_report = profiling_util::ReportBinding();
  std::cout << binding_report;
#endif

#ifdef _MPI
    MPI_Finalize();
#endif

    // Message for ReFrame sanity test to search for
    std::cout << "Affinity report finished!" << std::endl;

    return 0;
}
