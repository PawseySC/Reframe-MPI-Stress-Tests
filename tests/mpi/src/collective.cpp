#include "profile_util.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

#define Rank0LocalLogger() if (ThisTask==0) Log()
#define LogMPITest() Rank0LocalLogger()<<" running "<<mpifunc<< " test"<<std::endl;
#define LogMPIBroadcaster() if (ThisTask == itask) Log()<<" running "<<mpifunc<<" broadcasting "<<sendsize<<" GB"<<std::endl;
#define LogMPISender() Log()<<" Running "<<mpifunc<<" sending "<<sendsize<<" GB"<<std::endl;
#define LogMPIReceiver() if (ThisTask == itask) Log()<<" running "<<mpifunc<<std::endl;
#define LogMPIAllComm() Rank0LocalLogger()<<" running "<<mpifunc<<" all "<<sendsize<<" GB"<<std::endl;
#define Rank0ReportMem() if (ThisTask==0) {LogMemUsage();LogSystemMem();}

// Structure for data points
// Each data holds a value and a rank
struct Data {
    double value = 0.0; // Value of datum (could change to [x,y,z] coord and do calcs on x)
    int rank = 0;       // Rank it corresponds to (based on value)
};

// Report the current time
std::string getcurtime() {
    auto now = std::chrono::system_clock::now();
    auto curtime = std::chrono::system_clock::to_time_t(now);
    std::string s(std::ctime(&curtime));
    s = s.substr(0, s.size() - 1);
    return s;
}

// Global variables relating to the MPI communicator and world
// Rank number, world, size, and root rank
int ThisTask, NProcs;
int root_rank = 0;

template <typename Clock>
void reportTime(std::chrono::time_point<Clock> start, std::chrono::time_point<Clock> end, std::string func,
                std::string line) {
    float duration, duration_sum;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Declare a vector to hold the times for each rank
    std::vector<float> times(NProcs);
    void *p = times.data();
    // Gather all the times into `times` vector on root rank
    MPI_Gather(&duration, 1, MPI_FLOAT, p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Sum all the times and store in `duration_sum` on root rank
    MPI_Reduce(&duration, &duration_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only do the computation of time statistics on root rank
    if (ThisTask == root_rank) {
        auto avg = duration_sum / NProcs;
        float std = 0.0, iqr = 0.0;
        std::sort(times.begin(), times.end());
        auto min = times[0], max = times[0];
        // Calculate the inter-quartile range
        float q3 = (3 * (NProcs + 1)) / 4;
        float q1 = (NProcs + 1) / 4;
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
        std = sqrt((std / (NProcs - 1)));

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
    // Start timer here
    double duration = 0.0;
    duration -= MPI_Wtime();

    Rank0LocalLogger() << "Populating data vector on root rank" << std::endl;

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
    Rank0LocalLogger() << "Time taken to populate data vector on root rank: " << duration << " seconds" << std::endl;
}

// Broadcast data from root rank to all other ranks
void broadcastData(std::vector<Data> &data) {
    Rank0LocalLogger() << "Broadcasting data from root rank to all ranks" << std::endl;
    
    // Time the broadcast
    auto start = std::chrono::steady_clock::now();
    // Synchronise all ranks
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    MPI_Bcast(data.data(), data.size() * sizeof(Data), MPI_BYTE, root_rank, MPI_COMM_WORLD);
    // Synchronise all ranks after broadcast
    // MPI_Barrier(MPI_COMM_WORLD);

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    Rank0LocalLogger() << "Data broadcast from root rank to all ranks" << std::endl;
    reportTime(start, end, __func__, std::to_string(__LINE__));
}

// Redistribute data among all ranks using MPI collective communcation calls
// Uses Alltoall_v to redistribute data across all ranks
void redistributeData(std::vector<Data> &data, int ndata, int iverbose) {
    Rank0LocalLogger() << "Redistributing data" << std::endl;
    
    // Record the time taken
    auto start = std::chrono::steady_clock::now();
    // Vectors to hold the send and receive counts for each rank
    // nsend[i]_{rank j} is the number of data rank j sends to rank i
    // nrecv[i]_{rank j} is the number of data rank j receives from rank i
    std::vector<int> nsend(NProcs, 0), nrecv(NProcs, 0);
    int total_send = 0;
    // Array of send counts (in bytes) for each rank
    // send_counts[i]_{rank j} is the number of bytes rank j sends to rank i
    // int send_counts[NProcs] = {};
    std::vector<int> send_counts(NProcs, 0);
    // Populate the send vectors/arrays and the total send count
    for (auto i = 0; i < ndata; i++) {
        int irank = data[i].rank;
        if (irank != ThisTask) {
            nsend[irank]++;
            total_send++;
            send_counts[irank] = nsend[irank] * sizeof(Data);
        }
    }
    Log() << "Rank " << ThisTask << ": Sending " << total_send << " amount of data, requiring " << static_cast<double>(sizeof(Data) * total_send) / 1024. / 1024. / 1024. << "GB" << std::endl;

    // Vector to hold the data to send to all other ranks
    std::vector<Data> send_buffer;

    // Fill `send_buffer` with all data in `data` that do not belong on this rank
    // Uses a lambda to filter out all data that belong on this rank
    std::remove_copy_if(data.begin(), data.end(), std::back_inserter(send_buffer),
                        [](const Data &datum) { return datum.rank == ThisTask; });
    // Array to hold the offsets (relative to the start of `send_buffer`)
    // where each rank's data begins
    // int send_offsets[NProcs] {};
    std::vector<int> send_offsets(NProcs, 0);
    // Fill `send_offsets` by calculating the
    // cumulative sum of `send_counts`
    int total = 0;
    for (auto i = 1; i < NProcs; i++) {
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
    MPI_Allreduce(p1, p2, NProcs, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD);
    // Set `p1` and `p2` to nullptr
    p1 = nullptr;
    p2 = nullptr;
    // Each rank receives the same amount of data from all other ranks
    // This will need to be adjusted if I use transformData() and MPI_Allgather to fill `nrecv`
    int total_recv = nrecv[ThisTask] * (NProcs - 1);
    // Vector to hold all the data this rank receives
    std::vector<Data> recv_buffer(total_recv);
    // Number of bytes received from every rank
    // int recv_counts[NProcs] = {};
    std::vector<int> recv_counts(NProcs, 0);
    // Offsets (relative to start of `recv_buffer`)
    // where the data received from each rank is placed
    // int recv_offsets[NProcs] {};
    std::vector<int> recv_offsets(NProcs, 0);
    // Fill `recv_counts` with the entries from `nrecv` and
    // fill `recv_offsets` using a cumulative sum of `recv_counts`
    total = 0;
    for (auto i = 0; i < NProcs; i++) {
        if (i != ThisTask) {
            recv_counts[i] = nrecv[ThisTask];
        }
        recv_counts[i] *= sizeof(Data);
        if (i > 0) {
            total += recv_counts[i - 1];
            recv_offsets[i] = total;
        }
    }

    // Output the total no. of sends and receives for each rank
    if (iverbose) {
	Log() << "Rank " << ThisTask << ": MPI Communication has [total_send, total_recv] = " << total_send << ", " << total_recv << std::endl;
    }

    // Send data across all ranks using `MPI_Alltoallv`
    void *s = send_buffer.data();
    void *r = recv_buffer.data();
    int *sc = &send_counts[0];
    int *so = &send_offsets[0];
    int *rc = &recv_counts[0];
    int *ro = &recv_offsets[0];
    // MPI_Alltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype
    // datatype,
    //               void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype
    //               recvtype, MPI_Comm comm)
    MPI_Alltoallv(s, sc, so, MPI_BYTE, r, rc, ro, MPI_BYTE, MPI_COMM_WORLD);
    // Set `s` and `r` to null pointers
    s = nullptr;
    r = nullptr;

    // Fill `data`, replacing sent values with received ones,
    // perhaps expanding as well, adjusting `ndata` to `nlocal`
    auto nlocal = ndata - (total_send - total_recv);
    // Remove all entries from `data` that have been sent to other ranks
    data.erase(std::remove_if(data.begin(), data.end(), [](const Data &datum) { return datum.rank != ThisTask; }),
               data.end());
    // Resize `data` so it can hold all the entries
    // it has received from other ranks
    data.resize(nlocal);
    // `data` now has `nrecv[ThisTask]` non-zero entries, and
    // `nlocal - nrecv[ThisTask]` zero entries, which need to be filled
    auto idx = nrecv[ThisTask];
    for (auto i = 0; i < total_recv; i++) {
        data[idx++] = recv_buffer[i];
    }
    Log() << "Rank " << ThisTask << ": Now has " << nlocal << " elements" << std::endl;

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));
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
    Rank0LocalLogger() << "Gathering all data back onto root rank" << std::endl;
    
    // Reduce `data` vector down to its unique elements
    getUniqueElemenets(data);

    // Record time taken
    auto start = std::chrono::steady_clock::now();
    // Get total count of data on all ranks to declare receive buffer on root rank
    int nlocal = data.size();
    int ntotal;
    int *nlocal_list = (int *)malloc(sizeof(int) * NProcs);
    // Gather the number of data on each rank into one array that all ranks can view
    // This is used to compute the receive counts and offsets for sending the data
    // MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
    //               void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
    MPI_Allgather(&nlocal, 1, MPI_INT, nlocal_list, 1, MPI_INT, MPI_COMM_WORLD);
    // Compute total amount of data that will end up on the root rank
    // This will be the size of the receive buffer
    // MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op
    // op, MPI_Comm comm)
    MPI_Allreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);

    // Size of receive buffer for root rank
    // Will hold all data except what is already on root rank
    auto buffer_size = ntotal - nlocal_list[root_rank];
    std::vector<Data> recv_buffer(buffer_size);
    // Number of bytes received from every rank
    // int recv_counts[NProcs] = {};
    std::vector<int> recv_counts(NProcs, 0);
    // Offsets (relative to start of `recv_buffer`)
    // where the data received from each rank is placed
    // int recv_offsets[NProcs] {};
    std::vector<int> recv_offsets(NProcs, 0);
    // Fill `recv_counts` with the entries from `nlocal_list` and
    // fill `recv_offsets` using a cumulative sum of `recv_counts`
    int total = 0;
    for (auto i = 0; i < NProcs; i++) {
        if (i != ThisTask) {
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
    // MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
    //             void* recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
    //             int root, MPI_Comm comm)
    MPI_Gatherv(data.data(), send_bytes, MPI_BYTE, recv_buffer.data(), rc, ro, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Resize and fill out the rest of `data`
    if (ThisTask == root_rank) {
        data.resize(ntotal);
        for (auto i = 0; i < buffer_size; i++) {
            data[nlocal++] = recv_buffer[i];
        }
        if (iverbose) {
	    Log() << "Rank " << ThisTask << ": Now has " << nlocal << " elements" << std::endl;
        }
    }

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));
}

// Generate the data locally for each rank
std::vector<Data> generateData(int nlocal, float umin, float umax, float segment_width, bool iverbose) {
    
    Rank0LocalLogger() << "Generating data on all ranks" << std::endl;
    
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
        auto seed = rand() * ThisTask; // Seed random number generator
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
	    Log() << "Rank " << ThisTask << ": Has generated data of length " << nlocal << std::endl;
        }
#ifdef _OMP
    }
#endif

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    return data;
}

// Transform data by adding random Gaussian noise to each datum
void transformData(std::vector<Data> &data, int ndata, float segment_width) {
    // Record the time taken
    auto start = std::chrono::steady_clock::now();

    Rank0LocalLogger() << "Transforming data" << std::endl;

    // use OMP parallel region to apply Gaussian noise
#ifdef _OMP
#pragma omp parallel
    {
#endif
        // Generate random numbers for sd of Gaussian
        srand(time(NULL));
        auto seed = rand() * ThisTask; // Seed random number generator
        std::default_random_engine generator(seed);
        // Draw sd from uniform distribution between (0,1)
        std::uniform_real_distribution<double> udist(0, 1);
        auto sd = udist(generator);
        std::normal_distribution<double> ndist(0, sd);

#ifdef _OMP
#pragma omp for
#endif
        for (auto i = 0; i < ndata; i++) {
            // Force sd to be between (-1, 1)
            auto noise = std::fmod(ndist(generator), 1.0);
            double tmp = data[i].value + noise;
            int new_rank = static_cast<int>(std::floor(tmp / segment_width));
            // Force datum back into range (0,1) if the
            // Gaussian noise has pushed it outside
            if (new_rank >= NProcs) // datum > 1
            {
                data[i].value = tmp - 1;
                data[i].rank = new_rank - NProcs;
            } else if (new_rank < 0) // datum < 0
            {
                data[i].value = tmp + 1;
                data[i].rank = NProcs + new_rank;
            } else // 0 < datum < 1
            {
                data[i].value = tmp;
                data[i].rank = new_rank;
            }
        }
#ifdef _OMP
    }
#endif

    // Report the time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));
}

// Redistribute data among all ranks using MPI collective communcation calls
// Uses Alltoall_v to redistribute data across all ranks
void redistributeData2(std::vector<Data> &data, int &nlocal, int iverbose) {
    
    Rank0LocalLogger() << "Redistributing data" << std::endl;
    
    // Record the time taken
    auto start = std::chrono::steady_clock::now();
    // Vectors to hold the send and receive counts for each rank
    // nsend[i]_{rank j} is the number of data rank j sends to rank i
    // nrecv[i]_{rank j} is the number of data rank j receives from rank i
    std::vector<int> nsend(NProcs, 0), nrecv(NProcs * NProcs, 0);
    int total_send = 0;
    // Array of send counts (in bytes) for each rank
    // send_counts[i]_{rank j} is the number of bytes rank j sends to rank i
    // int send_counts[NProcs] = {};
    std::vector<int> send_counts(NProcs, 0);
    // Populate the send vectors/arrays and the total send count
    for (auto i = 0; i < nlocal; i++) {
        int irank = data[i].rank;
        if (irank != ThisTask) {
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
                        [](const Data &datum) { return datum.rank == ThisTask; });
    // Array to hold the offsets (relative to the start of `send_buffer`)
    // where each rank's data begins
    // int send_offsets[NProcs] {};
    std::vector<int> send_offsets(NProcs, 0);
    // Fill `send_offsets` by calculating the
    // cumulative sum of `send_counts`
    int total = 0;
    for (auto i = 1; i < NProcs; i++) {
        total += send_counts[i - 1];
        send_offsets[i] = total;
    }

    // Pointers for `nsend` and `nrecv` vectors
    auto p1 = nsend.data();
    auto p2 = nrecv.data();
    // Allgather is needed so each rank has a record of how many data
    // it receives from every other rank
    // MPI_Allgather(const void* sendbuf, int, sendcount, MPI_Datatype sendtype, void* recvbuf,
    //               int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
    MPI_Allgather(p1, NProcs, MPI_INTEGER, p2, NProcs, MPI_INTEGER, MPI_COMM_WORLD);

    int total_recv = 0;
    for (auto i = 0; i < NProcs; i++) {
        total_recv += nrecv[ThisTask + (i * NProcs)];
    }
    // Vector to hold all the data this rank receives
    std::vector<Data> recv_buffer(total_recv);
    // Number of bytes received from every rank
    // int recv_counts[NProcs] = {};
    std::vector<int> recv_counts(NProcs, 0);
    // Offsets (relative to start of `recv_buffer`)
    // where the data received from each rank is placed
    // int recv_offsets[NProcs] {};
    std::vector<int> recv_offsets(NProcs, 0);
    // Fill `recv_counts` with the entries from `nrecv` and
    // fill `recv_offsets` using a cumulative sum of `recv_counts`
    total = 0;
    for (auto i = 0; i < NProcs; i++) {
        if (i != ThisTask) {
            recv_counts[i] = nrecv[ThisTask + (i * NProcs)] * sizeof(Data);
        }
        if (i > 0) {
            total += recv_counts[i - 1];
            recv_offsets[i] = total;
        }
    }

    // Output the total no. of sends and receives for each rank
    if (iverbose) {
	Log() << "Rank " << ThisTask << ": MPI Communication has [total_send, total_recv] = " << total_send << ", " << total_recv << std::endl;
    }

    // Send data across all ranks using `MPI_Alltoallv`
    void *s = send_buffer.data();
    void *r = recv_buffer.data();
    int *sc = &send_counts[0];
    int *so = &send_offsets[0];
    int *rc = &recv_counts[0];
    int *ro = &recv_offsets[0];
    // MPI_Alltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype
    // datatype,
    //               void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype
    //               recvtype, MPI_Comm comm)
    MPI_Alltoallv(s, sc, so, MPI_BYTE, r, rc, ro, MPI_BYTE, MPI_COMM_WORLD);
    // Set `s` and `r` to null pointers
    s = nullptr;
    r = nullptr;

    // Fill `data`, replacing sent values with received ones,
    // perhaps expanding as well, adjusting `ndata` to `nlocal`
    auto new_nlocal = nlocal - (total_send - total_recv);
    // Remove all entries from `data` that have been sent to other ranks
    data.erase(std::remove_if(data.begin(), data.end(), [](const Data &datum) { return datum.rank != ThisTask; }),
               data.end());
    // Resize `data` so it can hold all the entries
    // it has received from other ranks
    data.resize(new_nlocal);
    // `data` now has `nrecv[ThisTask]` non-zero entries, and
    // `nlocal - nrecv[ThisTask]` zero entries, which need to be filled
    auto idx = nlocal - total_send;
    for (auto i = 0; i < total_recv; i++) {
        data[idx++] = recv_buffer[i];
    }
    if (iverbose) {
	Log() << "Rank " << ThisTask << ": Now has " << new_nlocal << " elements " << std::endl;
    }
    nlocal = new_nlocal;

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));
}

// Reduce the data that has been distributed among the
// ranks back onto the root rank as one datum
void reduceData(std::vector<Data> &data, int ndata, int iverbose) {

    Rank0LocalLogger() << ": Reducing data into summary statistic on root rank" << std::endl;
    
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
    // MPI_Reduce(const void* sendbuf, void*recvbuf, int count, MPI_Datatype datatype,
    //            MPI_Op op, int root, MPI_Comm comm)
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, root_rank, MPI_COMM_WORLD);

    if (iverbose)
    {
	Rank0LocalLogger() << "Global sum of all data: " << global_sum << std::endl;
    }

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));
}

// TODO: add call to MPI binding
int main(int argc, char *argv[]) {
    // Initial MPI setup
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    // Ensure program is called with right number of arguments
    if (ThisTask == root_rank) {
        if (argc < 3) {
            std::cout << "Usage: bcast.out ndata iverbose" << std::endl;
            return 1;
        }
    }
    // Ensure program is called with at least 2 MPI ranks
    if (NProcs == 1) {
        std::cout << "Error: This program was must be run with at least 2 MPI ranks" << std::endl;
        return 1;
    }

    // Number of data on each rank is `ndata=N^3`
    // So total data count is `NProcs * ndata = NProcs * N^3` where user passes N
    int ndata = pow(atoi(argv[1]), 3);
    int iverbose = atoi(argv[2]);

    // Bounds of distribution used to generate data entries
    float umin = 0.0;
    float umax = 1.0;
    // Width of data region for each rank
    float segment_width = (umax - umin) / NProcs;

    // Generate data that will be passed between the ranks
    MPI_Barrier(MPI_COMM_WORLD); // Synchronise tasks
    std::vector<Data> orig_data(ndata);
    // Generate the data on root rank
    if (ThisTask == root_rank) {
        populateData(ndata, orig_data, umin, umax, segment_width);
        // Sort the data by ascending value, using a lambda
        // std::sort(data.begin(), data.end(),
        //           [](const Data &a, const Data &b) {return a.value < b.value;});
        if (iverbose)
	{
	    Rank0LocalLogger() << "generated " << ndata << " elements" << std::endl;
        }
    }
    // Broadcast the data to all other ranks
    broadcastData(orig_data);

    // Redistribute data such that each rank holds
    // only the data that should be on that rank
    redistributeData(orig_data, ndata, iverbose);

    // Gather the data back to the root rank
    gatherData(orig_data, iverbose);

    // Delete all elements from `data` and clear memory
    orig_data.clear();
    orig_data.shrink_to_fit();

    // Get number of entries for each rank given
    // total data size and no. of ranks
    int nlocal = ndata / NProcs;
    // Handle when nprocs doesn't evenly divide into ndata
    // by assigning leftover data to final rank
    if (ThisTask == NProcs - 1) {
        nlocal = ndata - nlocal * (NProcs - 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto new_data = generateData(nlocal, umin, umax, segment_width, iverbose);

    // Transform the data on each rank to make them distinct
    // No longer strictly necessary, given my new way of
    // generating data for this second part
    // transformData(new_data, ndata, segment_width);

    // Redistribute data such that each rank holds
    // only the data that should be on that rank
    redistributeData2(new_data, nlocal, iverbose);

    // Do a computation and reduce on data such that
    // there is one value on root rank
    reduceData(new_data, ndata, iverbose);

    // Message for ReFrame test to make sure that job finished
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (ThisTask == root_rank) {
        std::cout << "Job completed at " << std::ctime(&end_time) << std::endl;
    }

    MPI_Finalize();
    return 0;
}
