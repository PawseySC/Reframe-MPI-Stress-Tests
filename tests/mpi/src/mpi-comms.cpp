#include "profile_util.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdio.h>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

// Prepend log statements with the rank being called as well as function and line number of calling statement
#define LOG(ThisTask)                                                                                                  \
    std::cout << _MPI_calling_rank(ThisTask) << "@" << __func__ << " L" << __LINE__ << " (" << getcurtime() << ") : "

// Structure for data points
// Each data holds a value and a rank
struct Data {
    double value = 0.0; // Value of datum (could change to [x,y,z] coord and do calcs on x)
    int rank = 0;       // Rank it corresponds to (based on value)
};

// Report the current
std::string getcurtime() {
    auto now = std::chrono::system_clock::now();
    auto curtime = std::chrono::system_clock::to_time_t(now);
    std::string s(std::ctime(&curtime));
    s = s.substr(0, s.size() - 1);
    return s;
}

// Global variables relating to the MPI communicator and world
// Rank number, world, size, and root rank
int world_rank, world_size;
int root_rank = 0;
char hostname[1024];

// Reports the running processes in this program
void reportProcs() {
    const char *proc_file = "/proc/self/status";
    std::string pid;
    std::ifstream f(proc_file);
    std::ofstream outfile;
    outfile.open("procs_list.txt", std::ios::out | std::ios::app);
    if (!f.is_open()) {
        std::cerr << "Couldn't open " << proc_file << " for proc register reading" << std::endl;
    }
    for (std::string line; std::getline(f, line);) {
        auto start = line.find("Pid:");
        if (start != std::string::npos) {
            std::istringstream is(line.substr(start + 5));
            is >> pid;
            // std::cout << "The PID for this process is: " << pid << " and it is running on node "
            //<< hostname << std::endl;
            outfile << "The PID for this process is: " << pid << " and it is running on node " << hostname << std::endl;
            break;
        }
    }
    outfile.close();
    f.close();
}

// Wrapper function for `profile_util` proc memory reporting
void logMemUsage(std::string func, std::string line_num) {
    std::string mem_usage_report = profiling_util::ReportMemUsage(func, line_num);
    std::cout << hostname << " : " << mem_usage_report << "." << std::endl;
}
// Wrapper function for `profile_util` system memory reporting
void logSystemMem(std::string func, std::string line_num) {
    std::string system_mem_report = profiling_util::ReportSystemMem(func, line_num);
    std::cout << hostname << " : " << system_mem_report << "." << std::endl;
}

// Wrapper functions for MPI functions
void my_MPIBcast(void *buffer, int count, MPI_Datatype datatype, int root) {
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    MPI_Bcast(buffer, count, datatype, root, MPI_COMM_WORLD);

    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}
void my_MPIAllreduce(const void *send_buf, void *recv_buf, int count, MPI_Datatype datatype, MPI_Op op) {
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
    //               MPI_Comm comm)
    MPI_Allreduce(send_buf, recv_buf, count, datatype, op, MPI_COMM_WORLD);

    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}
void my_MPIAlltoallv(const void *send_buf, const int send_counts[], const int send_offsets[], MPI_Datatype send_type,
                     void *recv_buf, const int recv_counts[], const int recv_offsets[], MPI_Datatype recv_type) {
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // MPI_Alltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype datatype,
    //               void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype,
    //               MPI_Comm comm)
    MPI_Alltoallv(send_buf, send_counts, send_offsets, send_type, recv_buf, recv_counts, recv_offsets, recv_type,
                  MPI_COMM_WORLD);

    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}
void my_MPIAllgather(const void *send_buf, int send_count, MPI_Datatype send_type, void *recv_buf, int recv_count,
                     MPI_Datatype recv_type) {
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
    //               void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
    MPI_Allgather(send_buf, send_count, send_type, recv_buf, recv_count, recv_type, MPI_COMM_WORLD);

    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}
void my_MPIGatherv(const void *send_buf, int send_count, MPI_Datatype send_type, void *recv_buf,
                   const int recv_counts[], const int recv_offsets[], MPI_Datatype recv_type, int root) {
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
    //             void* recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
    //             int root, MPI_Comm comm)
    MPI_Gatherv(send_buf, send_count, send_type, recv_buf, recv_counts, recv_offsets, recv_type, 0, MPI_COMM_WORLD);

    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}
void my_MPIReduce(const void *send_buf, void *recv_buf, int count, MPI_Datatype datatype, MPI_Op op, int root) {
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // MPI_Reduce(const void* sendbuf, void*recvbuf, int count, MPI_Datatype datatype,
    //            MPI_Op op, int root, MPI_Comm comm)
    MPI_Reduce(send_buf, recv_buf, count, datatype, op, root, MPI_COMM_WORLD);

    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}
void my_MPIScatter(const void *send_buf, int send_count, MPI_Datatype send_type, void *recv_buf, int recv_count,
                   MPI_Datatype recv_type, int root) {
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // MPI_Scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
    //             MPI_Datatype recvtype, int root, MPI_Comm comm)
    MPI_Scatter(send_buf, send_count, send_type, recv_buf, recv_count, recv_type, root, MPI_COMM_WORLD);

    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}
void my_MPIScatterv(const void *send_buf, const int send_counts[], const int send_offsets[], MPI_Datatype send_type,
                    void *recv_buf, int recv_count, MPI_Datatype recv_type, int root) {
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // MPI_Scatterv(const void* sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype,
    //              void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
    MPI_Scatterv(send_buf, send_counts, send_offsets, send_type, recv_buf, recv_count, recv_type, root, MPI_COMM_WORLD);

    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}

template <typename Clock>
void reportTime(std::chrono::time_point<Clock> start, std::chrono::time_point<Clock> end, std::string func,
                std::string line) {
    float duration, duration_sum;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Declare a vector to hold the times for each rank
    std::vector<float> times(world_size);
    void *p = times.data();
    // Gather all the times into `times` vector on root rank
    MPI_Gather(&duration, 1, MPI_FLOAT, p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Sum all the times and store in `duration_sum` on root rank
    MPI_Reduce(&duration, &duration_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only do the computation of time statistics on root rank
    if (world_rank == root_rank) {
        auto avg = duration_sum / world_size;
        float std = 0.0, iqr = 0.0;
        std::sort(times.begin(), times.end());
        auto min = times[0], max = times[0];
        // Calculate the inter-quartile range
        float q3 = (3 * (world_size + 1)) / 4;
        float q1 = (world_size + 1) / 4;
        float q3_term, q1_term;
        if (std::fmod(q3, 4.0) != 0) {
            q3_term = (times[static_cast<int>(std::floor(q3))] + times[static_cast<int>(std::ceil(q3))]) / 2;
        } else {
            q3_term = times[static_cast<int>(q3)];
        }
        if (std::fmod(q1, 4.0) != 0) {
            q1_term = (times[static_cast<int>(std::floor(q1))] + times[static_cast<int>(std::ceil(q1))]) / 2;
        } else {
            q1_term = times[static_cast<int>(q1)];
        }
        iqr = q3_term - q1_term;
        for (auto &time : times) {
            std += (time - avg) * (time - avg);
            min = std::min(time, min);
            max = std::max(time, max);
        }
        std = sqrt((std / (world_size - 1)));

        // Report the timing stats
        std::cout << "@" << func << ": L" << line << " time stats: ["
                  << "Average = " << avg << ", Standard Deviation = " << std << ", Minimum = " << min
                  << ", Maximum = " << max << ", IQR = " << iqr << ",] (us)" << std::endl;
    }

    // Synchronise all ranks after time reporting
    MPI_Barrier(MPI_COMM_WORLD);
}

// Populate data vector with random entries
void populateData(int ndata, std::vector<Data> &data, float umin, float umax, float segment_width) {
    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    // Start timer here
    double duration = 0.0;
    duration -= MPI_Wtime();
    if (world_rank == root_rank) {
        LOG(world_rank) << "Populating data vector on root rank" << std::endl;
    }

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
        std::uniform_real_distribution<double> udist(umin, umax);

#ifdef _OMP
#pragma omp for
#endif
        for (auto i = 0; i < ndata; i++) {
            data[i].value = udist(generator);
            data[i].rank = static_cast<int>(std::floor(data[i].value / segment_width));
        }
#ifdef _OMP
    }
#endif

    // Report time taken to populate data
    duration += MPI_Wtime();
    std::cout << "@" << __func__ << ": Time taken to populate data vector on root rank (" << root_rank
              << "): " << duration << " seconds" << std::endl;
    // Report memory
    if (world_rank == root_rank) {
        LOG(world_rank) << "Time taken to populate data vector on root rank: " << duration << " seconds" << std::endl;
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}

// Broadcast data from root rank to all other ranks
void broadcastData(std::vector<Data> &data) {
    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    if (world_rank == root_rank) {
        LOG(world_rank) << "Broadcasting data from root rank to all ranks" << std::endl;
    }
    // Time the broadcast
    auto start = std::chrono::steady_clock::now();
    // Synchronise all ranks
    // MPI_Barrier(MPI_COMM_WORLD);
    my_MPIBcast(data.data(), data.size() * sizeof(Data), MPI_BYTE, root_rank);
    // Synchronise all ranks after broadcast
    // MPI_Barrier(MPI_COMM_WORLD);

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}

// Redistribute data among all ranks using MPI collective communcation calls
// Uses Alltoall_v to redistribute data across all ranks
void redistributeData(std::vector<Data> &data, int ndata, int iverbose) {
    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    if (world_rank == root_rank) {
        LOG(world_rank) << "Redistributing data" << std::endl;
    }
    // Record the time taken
    auto start = std::chrono::steady_clock::now();
    // Vectors to hold the send and receive counts for each rank
    // nsend[i]_{rank j} is the number of data rank j sends to rank i
    // nrecv[i]_{rank j} is the number of data rank j receives from rank i
    std::vector<int> nsend(world_size, 0), nrecv(world_size, 0);
    int total_send = 0;
    // Vector of send counts (in bytes) for each rank
    // send_counts[i]_{rank j} is the number of bytes rank j sends to rank i
    std::vector<int> send_counts(world_size, 0);
    // Populate the send vectors/arrays and the total send count
    for (auto i = 0; i < ndata; i++) {
        int irank = data[i].rank;
        if (irank != world_rank) {
            nsend[irank]++;
            total_send++;
            send_counts[irank] = nsend[irank] * sizeof(Data);
        }
    }

    // Vector to hold the data to send to all other ranks
    std::vector<Data> send_buffer;

    // Fill `send_buffer` with all data in `data` that do not belong on this rank
    // Uses a lambda to filter out all data that belong on this rank
    std::remove_copy_if(data.begin(), data.end(), std::back_inserter(send_buffer),
                        [](const Data &datum) { return datum.rank == world_rank; });
    // Vector to hold the offsets (relative to the start of `send_buffer`) where each rank's data begins
    std::vector<int> send_offsets(world_size, 0);
    // Fill `send_offsets` by calculating the
    // cumulative sum of `send_counts`
    int total = 0;
    for (auto i = 1; i < world_size; i++) {
        total += send_counts[i - 1];
        send_offsets[i] = total;
    }

    // Pointers for `nsend` and `nrecv` vectors
    auto p1 = nsend.data();
    auto p2 = nrecv.data();
    // Populate `nrecv` using `MPI_ALLreduce` and the `nsend` vectors from all ranks
    // Every rank sends the same amount of data to each rank, so can use Allreduce with MPI_MAX
    // Once I add the transformData() function to make each rank's data unique
    // then I will need to change this (probably to what I use in pt2pt)
    my_MPIAllreduce(p1, p2, world_size, MPI_INTEGER, MPI_MAX);
    // Set `p1` and `p2` to nullptr
    p1 = nullptr;
    p2 = nullptr;
    // Each rank receives the same amount of data from all other ranks
    // This will need to be adjusted if I use transformData() and MPI_Allgather to fill `nrecv`
    int total_recv = nrecv[world_rank] * (world_size - 1);
    // Vector to hold all the data this rank receives
    std::vector<Data> recv_buffer(total_recv);
    // Number of bytes received from every rank
    std::vector<int> recv_counts(world_size);
    // Offsets (relative to start of `recv_buffer`) where the data received from each rank is placed
    std::vector<int> recv_offsets(world_size);
    // Fill `recv_counts` with the entries from `nrecv` and
    // fill `recv_offsets` using a cumulative sum of `recv_counts`
    total = 0;
    for (auto i = 0; i < world_size; i++) {
        if (i != world_rank) {
            recv_counts[i] = nrecv[world_rank];
        }
        recv_counts[i] *= sizeof(Data);
        if (i > 0) {
            total += recv_counts[i - 1];
            recv_offsets[i] = total;
        }
    }

    // Output the total no. of sends and receives for each rank
    if (iverbose) {
        LOG(world_rank) << "Has [total_send, total_recv] = " << total_send << ", " << total_recv << std::endl;
    }

    // Send data across all ranks using `MPI_Alltoallv`
    // Convert vectors to appropriate format for MPI function calls (arrays/addresses vs. vectors)
    void *s = send_buffer.data();
    void *r = recv_buffer.data();
    int *sc = &send_counts[0];
    int *so = &send_offsets[0];
    int *rc = &recv_counts[0];
    int *ro = &recv_offsets[0];
    my_MPIAlltoallv(s, sc, so, MPI_BYTE, r, rc, ro, MPI_BYTE);
    // Set `s` and `r` to null pointers
    s = nullptr;
    r = nullptr;

    // Fill `data`, replacing sent values with received ones,
    // perhaps expanding as well, adjusting `ndata` to `nlocal`
    auto nlocal = ndata - (total_send - total_recv);
    // Remove all entries from `data` that have been sent to other ranks
    data.erase(std::remove_if(data.begin(), data.end(), [](const Data &datum) { return datum.rank != world_rank; }),
               data.end());
    // Resize `data` so it can hold all the entries
    // it has received from other ranks
    data.resize(nlocal);
    // `data` now has `nrecv[world_rank]` non-zero entries, and
    // `nlocal - nrecv[world_rank]` zero entries, which need to be filled
    auto idx = nrecv[world_rank];
    for (auto i = 0; i < total_recv; i++) {
        data[idx++] = recv_buffer[i];
    }
    if (iverbose) {
        LOG(world_rank) "Now has " << nlocal << " elements" << std::endl;
    }

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}

void getUniqueElemenets(std::vector<Data> &data) {
    // Sort `data` by data values
    std::sort(data.begin(), data.end(), [](const Data &a, const Data &b) { return a.value < b.value; });
    std::vector<Data>::iterator it;

    // Gather all unique elements to the front
    it = std::unique(data.begin(), data.end(), [](const Data &a, const Data &b) { return a.value == b.value; });

    // Resize vector (non-unique elements are pushed out)
    data.resize(std::distance(data.begin(), it));
}

void gatherData(std::vector<Data> &data, int iverbose) {
    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
        LOG(world_rank) << "Gathering all data back onto root rank" << std::endl;
    }
    // Reduce `data` vector down to its unique elements
    getUniqueElemenets(data);

    // Record time taken
    auto start = std::chrono::steady_clock::now();
    // Get total count of data on all ranks to declare receive buffer on root rank
    int nlocal = data.size();
    int ntotal;
    int *nlocal_list = (int *)malloc(sizeof(int) * world_size);
    // Gather the number of data on each rank into one array that all ranks can view
    // This is used to compute the receive counts and offsets for sending the data
    my_MPIAllgather(&nlocal, 1, MPI_INT, nlocal_list, 1, MPI_INT);
    // Compute total amount of data that will end up on the root rank
    // This will be the size of the receive buffer
    my_MPIAllreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM);
    // MPI_Barrier(MPI_COMM_WORLD);

    // Size of receive buffer for root rank
    // Will hold all data except what is already on root rank
    auto buffer_size = ntotal - nlocal_list[root_rank];
    std::vector<Data> recv_buffer(buffer_size);
    // Number of bytes received from every rank
    std::vector<int> recv_counts(world_size);
    // Offsets (relative to start of `recv_buffer`) where the data received from each rank is placed
    std::vector<int> recv_offsets(world_size);
    // Fill `recv_counts` with the entries from `nlocal_list` and
    // fill `recv_offsets` using a cumulative sum of `recv_counts`
    int total = 0;
    for (auto i = 0; i < world_size; i++) {
        if (i != world_rank) {
            recv_counts[i] = nlocal_list[i];
        }
        recv_counts[i] *= sizeof(Data);
        if (i > 0) {
            total += recv_counts[i - 1];
            recv_offsets[i] = total;
        }
    }

    // Total amount of bytes this rank will send
    auto send_bytes = nlocal * sizeof(Data);
    // Gather all the data from each rank onto the root rank
    int *rc = &recv_counts[0];
    int *ro = &recv_offsets[0];
    my_MPIGatherv(data.data(), send_bytes, MPI_BYTE, recv_buffer.data(), rc, ro, MPI_BYTE, 0);

    // Resize and fill out the rest of `data`
    if (world_rank == root_rank) {
        data.resize(ntotal);
        for (auto i = 0; i < buffer_size; i++) {
            data[nlocal++] = recv_buffer[i];
        }
        if (iverbose) {
            LOG(world_rank) << "Root rank now has " << nlocal << " elements" << std::endl;
        }
    }

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}

// Generate the data locally for each rank
std::vector<Data> generateData(int nlocal, float umin, float umax, float segment_width, bool iverbose) {
    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    if (world_rank == root_rank) {
        LOG(world_rank) << "Generating data on all ranks" << std::endl;
    }
    // Record time taken
    auto start = std::chrono::steady_clock::now();
    // Declare vector of data for each rank
    std::vector<Data> data(nlocal);

// Use OMP parallelisation for actual generation of data
#ifdef _OMP
#pragma omp parallel
    {
#endif
        // Generate random numbers for the vector entries
        srand(time(NULL));
        auto seed = rand() * world_rank; // Seed random number generator
        std::default_random_engine generator(seed);
        // Get range of uniform distribution - default is U(0,1)
        std::uniform_real_distribution<double> udist(umin, umax);

#ifdef _OMP
#pragma omp for
#endif
        for (auto i = 0; i < nlocal; i++) {
            data[i].value = udist(generator);
            data[i].rank = static_cast<int>(std::floor(data[i].value / segment_width));
        }

        if (iverbose) {
            LOG(world_rank) << "Has generated data of length " << nlocal << std::endl;
        }
#ifdef _OMP
    }
#endif

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }

    return data;
}

// Redistribute data among all ranks using MPI collective communcation calls
// Uses Alltoall_v to redistribute data across all ranks
void redistributeData2(std::vector<Data> &data, int &nlocal, int iverbose) {
    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
        LOG(world_rank) << "Redistributing data" << std::endl;
    }

    // Record the time taken
    auto start = std::chrono::steady_clock::now();
    // Vectors to hold the send and receive counts for each rank
    // nsend[i]_{rank j} is the number of data rank j sends to rank i
    // nrecv[i]_{rank j} is the number of data rank j receives from rank i
    std::vector<int> nsend(world_size, 0), nrecv(world_size * world_size, 0);
    int total_send = 0;
    // Vector of send counts (in bytes) for each rank
    // send_counts[i]_{rank j} is the number of bytes rank j sends to rank i
    std::vector<int> send_counts(world_size);
    // Populate the send vectors/arrays and the total send count
    for (auto i = 0; i < nlocal; i++) {
        int irank = data[i].rank;
        if (irank != world_rank) {
            nsend[irank]++;
            total_send++;
            send_counts[irank] = nsend[irank] * sizeof(Data);
        }
    }

    // Vector to hold the data to send to all other ranks
    std::vector<Data> send_buffer;

    // Fill `send_buffer` with all data in `data` that do not belong on this rank
    // Uses a lambda to filter out all data that belong on this rank
    std::remove_copy_if(data.begin(), data.end(), std::back_inserter(send_buffer),
                        [](const Data &datum) { return datum.rank == world_rank; });
    // Vector to hold the offsets (relative to the start of `send_buffer`) where each rank's data begins
    std::vector<int> send_offsets(world_size);
    // Fill `send_offsets` by calculating the
    // cumulative sum of `send_counts`
    int total = 0;
    for (auto i = 1; i < world_size; i++) {
        total += send_counts[i - 1];
        send_offsets[i] = total;
    }

    // Pointers for `nsend` and `nrecv` vectors
    auto p1 = nsend.data();
    auto p2 = nrecv.data();
    // Allgather is needed so each rank has a record of how many data
    // it receives from every other rank
    my_MPIAllgather(p1, world_size, MPI_INT, p2, world_size, MPI_INT);

    int total_recv = 0;
    for (auto i = 0; i < world_size; i++) {
        total_recv += nrecv[world_rank + (i * world_size)];
    }
    // Vector to hold all the data this rank receives
    std::vector<Data> recv_buffer(total_recv);
    // Number of bytes received from every rank
    std::vector<int> recv_counts(world_size);
    // Offsets (relative to start of `recv_buffer`)
    // where the data received from each rank is placed
    std::vector<int> recv_offsets(world_size);
    // Fill `recv_counts` with the entries from `nrecv` and
    // fill `recv_offsets` using a cumulative sum of `recv_counts`
    total = 0;
    for (auto i = 0; i < world_size; i++) {
        if (i != world_rank) {
            recv_counts[i] = nrecv[world_rank + (i * world_size)] * sizeof(Data);
        }
        if (i > 0) {
            total += recv_counts[i - 1];
            recv_offsets[i] = total;
        }
    }

    // Output the total no. of sends and receives for each rank
    if (iverbose) {
        LOG(world_rank) << "Has [total_send, total_recv] = " << total_send << ", " << total_recv << std::endl;
    }

    // Send data across all ranks using `MPI_Alltoallv`
    // Convert vectors to formats required by MPI function calls (addresses and arrays)
    void *s = send_buffer.data();
    void *r = recv_buffer.data();
    int *sc = &send_counts[0];
    int *so = &send_offsets[0];
    int *rc = &recv_counts[0];
    int *ro = &recv_offsets[0];
    my_MPIAlltoallv(s, sc, so, MPI_BYTE, r, rc, ro, MPI_BYTE);
    // Set `s` and `r` to null pointers
    s = nullptr;
    r = nullptr;

    // Fill `data`, replacing sent values with received ones,
    // perhaps expanding as well, adjusting `ndata` to `nlocal`
    auto new_nlocal = nlocal - (total_send - total_recv);
    // Remove all entries from `data` that have been sent to other ranks
    data.erase(std::remove_if(data.begin(), data.end(), [](const Data &datum) { return datum.rank != world_rank; }),
               data.end());
    // Resize `data` so it can hold all the entries
    // it has received from other ranks
    data.resize(new_nlocal);
    // `data` now has `nrecv[world_rank]` non-zero entries, and
    // `nlocal - nrecv[world_rank]` zero entries, which need to be filled
    auto idx = nlocal - total_send;
    for (auto i = 0; i < total_recv; i++) {
        data[idx++] = recv_buffer[i];
    }
    if (iverbose) {
        LOG(world_rank) << "Now has " << new_nlocal << " elements " << std::endl;
    }
    nlocal = new_nlocal;

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}

// Reduce the data that has been distributed among the
// ranks back onto the root rank as one datum
void reduceData(std::vector<Data> &data, int ndata, int iverbose) {
    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
        LOG(world_rank) << "Reducing data into summary statistic on root rank" << std::endl;
    }

    // Record time taken
    auto start = std::chrono::steady_clock::now();

    // Perform initial (arbitrary) computation on each rank to give a single number, since
    // MPI_Reduce needs each rank to pass the same number of data to the root
    // rank, and currently each rank has different number of data
    float local_sum, global_sum;
    for (auto &datum : data) {
        local_sum += datum.value;
    }

    // Reduce local sums into one global sum on the root rank
    my_MPIReduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, root_rank);

    if ((world_rank == root_rank) & (iverbose)) {
        LOG(world_rank) << "Global sum of all data: " << global_sum << std::endl;
    }

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}

// Need to add scatter and pt2pt
void scatterData(std::vector<Data> &data, int ndata, int iverbose) {
    if (world_rank == root_rank) {
        LOG(world_rank) << "Scattering data from root rank to all ranks" << std::endl;
    }
    // Record the time taken
    auto start = std::chrono::steady_clock::now();

    // Vector to hold the send and receive counts for each rank
    // nsend[i] is the number of data the root rank sends to rank i
    std::vector<int> nsend(world_size, 0);
    // `nrecv` is the number of data each rank receives from the root rank
    int *nrecv = (int *)malloc(sizeof(int) * 1);
    // Vector of send counts (in bytes) for each rank
    // send_counts[i]_{rank j} is the number of bytes rank j sends to rank i
    std::vector<int> send_counts(world_size);
    // Vector to hold the offsets (relative to the start of `send_buffer`) where each rank's data begins
    std::vector<int> send_offsets(world_size);
    // Populate the send vectors/arrays and the total send count
    if (world_rank == root_rank) {
        for (auto i = 0; i < ndata; i++) {
            int irank = data[i].rank;
            nsend[irank]++;
            send_counts[irank] = nsend[irank] * sizeof(Data);
        }

        // Fill `send_offsets` by calculating the
        // cumulative sum of `send_counts`
        int total = 0;
        for (auto i = 1; i < world_size; i++) {
            total += send_counts[i - 1];
            send_offsets[i] = total;
        }
    }
    // Send the `nrecv` values to each rank with MPI_Scatter (it goes into nrecv[0])
    my_MPIScatter(nsend.data(), 1, MPI_INT, nrecv, 1, MPI_INT, root_rank);

    // Number of bytes that each rank receives from the root rank
    int recv_count = nrecv[0] * sizeof(Data);
    // Vector to hold all the data this rank receives
    std::vector<Data> recv_buffer(nrecv[0]);

    // Output the total no. of data each rank receives from the root rank
    if (iverbose) {
        LOG(world_rank) << "Receiving " << nrecv[0] << " from root rank" << std::endl;
    }

    // Send data from root rank to all other ranks using 'MPI_Scatterv'
    void *s = data.data();
    void *r = recv_buffer.data();
    int *sc = &send_counts[0];
    int *so = &send_offsets[0];
    my_MPIScatterv(s, sc, so, MPI_BYTE, r, recv_count, MPI_BYTE, root_rank);
    s = nullptr;
    r = nullptr;

    // Fill `data`, with the data received from the root rank, resizing as necessary
    auto nlocal = nrecv[0];
    data.resize(nlocal);
    data.shrink_to_fit();
    for (auto i = 0; i < nlocal; i++) {
        data[i] = recv_buffer[i];
    }
    if (iverbose) {
        LOG(world_rank) << "Now has " << nlocal << " elements" << std::endl;
    }

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    // Report memory
    if (world_rank == root_rank) {
        logMemUsage(__func__, std::to_string(__LINE__));
        logSystemMem(__func__, std::to_string(__LINE__));
    }
}

#ifdef _MPI
inline std::string _gethostname() {
    char hnbuf[64];
    memset(hnbuf, 0, sizeof(hnbuf));
    (void)gethostname(hnbuf, sizeof(hnbuf));
    return std::string(hnbuf);
}
#endif

// TODO: add call to MPI binding
int main(int argc, char *argv[]) {
    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    // Initial MPI setup
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // hostname stuff
    // NEED TO GET UNIQUE HOSTNAMES FOR REFRAME TEST
    gethostname(hostname, 1024);
    // if (world_rank == root_rank)
    //{
    // std::cout << "Running on node " << hostname << std::endl;
    // }
    // printf("Rank %i of %i running on processor %i on %s.\n", rank, size, proc, hostname);
    //  auto hostname = _gethostname();
    //  int size = hostname.size() + 1, maxsize = 0;
    //  // MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
    //  //               MPI_Comm comm)
    //  MPI_Allreduce(&size, &maxsize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    //  std::vector<char> allhostnames(maxsize * world_size);
    //  for (auto i = size; i < maxsize; i++)
    //  {
    //      hostname += " ";
    //  }
    //  // MPI_Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
    //  //            int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
    //  MPI_Gather(hostname.c_str(), maxsize, MPI_CHAR, allhostnames.data(), maxsize, MPI_CHAR, 0, MPI_COMM_WORLD);
    //  std::unordered_set<std::string> hostnames;
    //  for (auto i = 0; i < world_size; i++)
    //  {
    //      std::string s;
    //      for (auto j = 0; j < maxsize; j ++)
    //      {
    //          s += allhostnames[i * maxsize + j];
    //          hostnames.insert(s);
    //      }
    //  }
    //  if (world_rank == root_rank)
    //  {
    //      std::cout << "Hostnames: ";
    //      for (auto i = 0; i < maxsize * world_size; i++)
    //      {
    //          std::cout << allhostnames[i];
    //      }
    //      std::cout << std::endl;
    //  }

    // Ensure program is called with right number of arguments
    if (world_rank == root_rank) {
        if (argc < 3) {
            std::cout << "Usage: mpi-comms.out ndata iverbose" << std::endl;
            return 1;
        }
        std::cout << "Memory sampling started at " << std::ctime(&start_time) << std::endl;
    }
    // Ensure program is called with at least 2 MPI ranks
    if (world_size == 1) {
        std::cout << "Error: This program was must be run with at least 2 MPI ranks" << std::endl;
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    reportProcs();
    MPI_Barrier(MPI_COMM_WORLD);
    // sleep(10);

    // Number of data on each rank is `ndata=N^3`
    // So total data count is `world_size * ndata = world_size * N^3` where user passes N
    int ndata = pow(atoi(argv[1]), 3);
    int iverbose = atoi(argv[2]);

    // Bounds of distribution used to generate data entries
    float umin = 0.0;
    float umax = 1.0;
    // Width of data region for each rank
    float segment_width = (umax - umin) / world_size;

    // Generate data that will be passed between the ranks
    MPI_Barrier(MPI_COMM_WORLD); // Synchronise tasks
    std::vector<Data> orig_data(ndata);
    // Generate the data on root rank
    if (world_rank == root_rank) {
        populateData(ndata, orig_data, umin, umax, segment_width);
        if (iverbose) {
            LOG(world_rank) << "Root rank generated " << ndata << " elements" << std::endl;
        }
        std::sort(orig_data.begin(), orig_data.end(), [](const Data &a, const Data &b) { return a.value < b.value; });
    }
    // Broadcast the data to all other ranks
    broadcastData(orig_data);

    // Redistribute data such that each rank holds
    // only the data that should be on that rank
    redistributeData(orig_data, ndata, iverbose);

    // Gather the data back to the root rank
    gatherData(orig_data, iverbose);

    // Delete all elements from `orig_data` and clear memory
    orig_data.clear();
    orig_data.shrink_to_fit();

    // Get number of entries for each rank given
    // total data size and no. of ranks
    int nlocal = ndata / world_size;
    // Handle when nprocs doesn't evenly divide into ndata
    // by assigning leftover data to final rank
    if (world_rank == world_size - 1) {
        nlocal = ndata - nlocal * (world_size - 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto new_data = generateData(nlocal, umin, umax, segment_width, iverbose);
    std::sort(new_data.begin(), new_data.end(), [](const Data &a, const Data &b) { return a.value < b.value; });

    // Redistribute data such that each rank holds
    // only the data that should be on that rank
    redistributeData2(new_data, nlocal, iverbose);

    // Do a computation and reduce on data such that
    // there is one value on root rank
    reduceData(new_data, ndata, iverbose);

    // Delete all elements from `new_data` and clear memory
    new_data.clear();
    new_data.shrink_to_fit();

    // Generate data that will be passed between the ranks
    MPI_Barrier(MPI_COMM_WORLD); // Synchronise tasks
    std::vector<Data> scattering_data(ndata);
    // Generate the data on root rank
    if (world_rank == root_rank) {
        populateData(ndata, scattering_data, umin, umax, segment_width);
        if (iverbose) {
            LOG(world_rank) << "Root rank generated " << ndata << " elements" << std::endl;
        }
        std::sort(scattering_data.begin(), scattering_data.end(),
                  [](const Data &a, const Data &b) { return a.value < b.value; });
    }
    // Scatter data to other ranks
    MPI_Barrier(MPI_COMM_WORLD);
    scatterData(scattering_data, ndata, iverbose);

    // // Delete file so that process listing program stops
    // if (world_rank == root_rank)
    // {
    //     std::cout << "REMOVING PROCS_LIST" << std::endl;
    //     const int result = remove("procs_list.txt");
    // }

    // Message for ReFrame test to make sure that job finished
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (world_rank == root_rank) {
        std::cout << "Memory sampling completed at " << std::ctime(&end_time) << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::ofstream outfile("done.txt");
    outfile << "DONE!" << std::endl;
    outfile.close();

    MPI_Finalize();
    return 0;
}