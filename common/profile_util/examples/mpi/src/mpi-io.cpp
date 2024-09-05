#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>
#include <thread>
#include <mpi.h>
#include <profile_util.h>

int ThisTask, NProcs;
std::chrono::system_clock::time_point logtime;
std::time_t log_time;
char wherebuff[1000];
std::string whenbuff;

#define Rank0LocalLogger() if (ThisTask==0) Log()<<wherebuff<<" : " 
#define LogMPITest() Rank0LocalLogger()<<" running "<<mpifunc<< " test"<<std::endl;
#define LogMPIBroadcaster() if (ThisTask == itask) Log()<<"Running "<<mpifunc<<" broadcasting "<<sendsize<<" GB"<<std::endl;
#define LogMPISender() Log()<<"Running "<<mpifunc<<" sending "<<sendsize<<" GB"<<std::endl;
#define LogMPIReceiver() if (ThisTask == itask) Log()<<"Running "<<mpifunc<<std::endl;
#define LogMPIAllComm() Rank0LocalLogger()<<"Running "<<mpifunc<<" all "<<sendsize<<" GB"<<std::endl;
#define Rank0ReportMem() if (ThisTask==0) {LogMemUsage();LogSystemMem();}

void WriteCollective(MPI_Comm &comm, std::string &fnamebase) {
    std::string fname=fnamebase+std::string(".collective.example.txt");
    Rank0LocalLogger()<<" Starting collective binary write to "<<fname<<std::endl;    
    MPI_File file;
    MPI_Offset offset;
    MPI_Status status;
    std::vector<int> buffer(100);
    for (auto i=0;i<buffer.size();i++) buffer[i] = ThisTask*NProcs + i;

    // Open the file in parallel
    MPI_File_open(comm, fname.c_str(), 
        MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, 
        &file);
    // Calculate the offset based on the rank
    offset = ThisTask * buffer.size()*sizeof(int);
    // Write to the file using collective I/O of binary data
    MPI_File_write_at_all(file, offset, buffer.data(), buffer.size(), MPI_INT, &status);
    // Close the file
    MPI_File_close(&file);
    Rank0LocalLogger()<<" Finished collective binary write "<<std::endl;    
}

void WriteNonCollective(MPI_Comm &comm, std::string &fnamebase) {
    std::string fname=fnamebase+std::string(".non-collective.example.txt");
    Rank0LocalLogger()<<" Starting non-collective binary write to "<<fname<<std::endl;
    MPI_File file;
    MPI_Offset offset;
    MPI_Status status;
    std::vector<int> buffer(100);
    for (auto i=0;i<buffer.size();i++) buffer[i] = ThisTask*NProcs + i;

    // Open the file in parallel
    MPI_File_open(comm, fname.c_str(), 
        MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, 
        &file);
    // Calculate the offset based on the rank
    offset = ThisTask * buffer.size()*sizeof(int);
    // Write to the file using non-collective I/O
    MPI_File_write_at(file, offset, buffer.data(), 50, MPI_INT, &status);
    // Write to the file again using non-collective I/O
    MPI_File_write_at(file, offset + sizeof(int)*50, buffer.data() + 50, 50, MPI_INT, &status);

    // Close the file
    MPI_File_close(&file);
    Rank0LocalLogger()<<" Finished non-collective binary write "<<std::endl;    
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    auto comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NProcs);
    MPISetLoggingComm(comm);
    std::string fname = "mpi-io";
    if (argc == 2) fname = std::string(argv[1]);

    // init logger time
    logtime = std::chrono::system_clock::now();
    log_time = std::chrono::system_clock::to_time_t(logtime);
    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    Rank0LocalLogger()<<"Starting job "<<std::endl;
    Rank0ReportMem();
    MPILog0NodeMemUsage();
    MPILog0NodeSystemMem();
    MPILog0ParallelAPI();
    MPILog0Binding();

    // trial some writes 
    WriteCollective(comm,fname);
    WriteNonCollective(comm,fname);

    // finish job
    Rank0LocalLogger()<<"Ending job "<<std::endl;
    MPI_Finalize();
    return 0;
}
