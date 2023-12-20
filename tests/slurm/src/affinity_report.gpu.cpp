#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>
#include <thread>
#include <profile_util.h>


#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MPI
#include <mpi.h>
#endif


int ThisTask, NProcs;
std::chrono::system_clock::time_point logtime;
std::time_t log_time;
char wherebuff[1000];
std::string whenbuff;

#define Where() sprintf(wherebuff,"[%04d] @%sL%d ", ThisTask,__func__, __LINE__);
#define When() logtime = std::chrono::system_clock::now(); log_time = std::chrono::system_clock::to_time_t(logtime);whenbuff=std::ctime(&log_time);whenbuff.erase(std::find(whenbuff.begin(), whenbuff.end(), '\n'), whenbuff.end());
#define LocalLogger() Where();std::cout<<wherebuff<<" : " 
#define Rank0LocalLogger() Where();if (ThisTask==0) std::cout<<wherebuff<<" : " 
#define LogMPIAllComm() Rank0LocalLoggerWithTime()<<" running "<<mpifunc<<" all "<<sendsize<<" GB"<<std::endl;




int main(int argc, char **argv) {

    char hostname[1024];
    int proc = sched_getcpu();
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &NProcs);
    MPI_Comm_rank(comm, &ThisTask);
    gethostname(hostname, 1024);
    printf("Rank %i of %i running on processor %i on %s.\n", ThisTask, NProcs, proc, hostname);

    MPILog0ParallelAPI();
#ifdef _OPENMP
    #pragma omp parallel
    {
        MPILogThreadAffinity(comm);
    }
#endif
    MPILog0Binding();

#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif

    return 0;
}
