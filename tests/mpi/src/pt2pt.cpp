#include "profile_util/include/profile_util.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

// Modes of point-to-point communication dictating the specifc MPI send and receive routines
// These numbers correspond to possible values of command-line argument `iblocking`
#define FULLBLOCKING 0    // blocking send and receives
#define PARTIALBLOCKING 1 // blocking send, non-blocking receives
#define ASYNCHWAITS 2     // asynchronous send and receives, with wait after each receive
#define ASYNCHRONOUS 3    // fully asynchronous send and receives, with waitall after all receives

// Prepend log statements with the rank being called as well as function and line number of calling statement
#define LOG(ThisTask)                                                                                                  \
    std::cout << _MPI_calling_rank(ThisTask) << "@" << __func__ << " L" << __LINE__ << " (" << getcurtime() << ") : "

// To ease redistribution of data, have each
// datum also have a rank indicator to denote
// which rank it is on
struct Data {
    double coord[3] = {0.0, 0.0, 0.0}; // Coordinate of datum
    int rank = 0;                      // Rank it is on
};

// Report the current time
std::string getcurtime() {
    auto now = std::chrono::system_clock::now();
    auto curtime = std::chrono::system_clock::to_time_t(now);
    std::string s(std::ctime(&curtime));
    s = s.substr(0, s.size() - 1);
    return s;
}

// This rank number and size of MPI_COMM_WORLD
int world_rank, world_size;
int root_rank = 0;

template <typename Clock>
void reportTime(std::chrono::time_point<Clock> start, std::chrono::time_point<Clock> end, std::string func,
                std::string line) {
    float duration, duration_sum;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Declare a vector to hold the times from each rank
    std::vector<float> times(world_size);
    void *p = times.data();
    // Gather all the times into `times` vector on rank 0
    MPI_Gather(&duration, 1, MPI_INT, p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Sum all the times and store in `duration_sum` on rank 0 for computation of average
    MPI_Reduce(&duration, &duration_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only do the computation and reporting on rakn 0
    if (world_rank == 0) {
        auto avg = duration_sum / world_size;
        float std = 0.0, iqr = 0.0;
        std::sort(times.begin(), times.end());
        auto min = times[0], max = times[0]; // min and max times
        // Calculate the inter-quartile range
        float q1 = (world_size + 1) / 4, q3 = (3 * (world_size + 1)) / 4;
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

// Generate the initial data locally for each rank
std::vector<Data> generateData(int nlocal, float umin, float umax, float segment_width, bool iadjacent) {
    if (world_rank == root_rank) {
        LOG(world_rank) << "Generating data on all ranks" << std::endl;
    }
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
        // Get range of unfirom distribution - default is U(0,1)
        if (iadjacent) {
            umin = std::max(umin, (world_rank - 1) * segment_width);
            umax = std::min(umax, (world_rank + 2) * segment_width);
        }
        std::uniform_real_distribution<double> udist(umin, umax);

#ifdef _OMP
#pragma omp for
#endif
        for (auto i = 0; i < nlocal; i++) {
            for (auto j = 0; j < 3; j++) {
                data[i].coord[j] = udist(generator);
            }
            data[i].rank = world_rank;
        }
#ifdef _OMP
    }
#endif
    LOG(world_rank) << "Has generated data with length " << nlocal << std::endl;

    // Report time taken
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));

    return data;
}

// Redistribute the data between the ranks
// Each rank can communicate with all other ranks
void redistributeData(std::vector<Data> &data, int nlocal, float umax, float umin, float segment_width, int iblocking,
                      bool iadjacent, bool iverbose, bool idelay, bool irandom) {
    auto comm = MPI_COMM_WORLD;
    auto ThisTask = world_rank;
    if (world_rank == root_rank) {
        LOG(world_rank) << "Redistributing data" << std::endl;
    }
    // Record time taken
    auto start = std::chrono::steady_clock::now();
    // Arrays to hold how many elements each rank sends and receives
    // `nsend[i]` is how many data this rank sends to rank `i`
    // `nrecv` has size `world_size^2` because each rank needs access to the
    // `nsend` of every rank, to determine how many data were sent to it by
    // each rank
    // So nrecv[0:world_size-1] = nsend_0, nrecv[world_size:2*world_size-1] = nsend_1, etc.
    std::vector<int> nsend(world_size, 0), nrecv(world_size * world_size, 0);
    int total_send = 0, total_recv = 0;

    // Divide space of uniform distribution used to generate data
    // into (approximately) equal segments of width (umax - umin) / world_size
    // All data values are sent to rank corresponding to their value
    // e.g. U(0, 1) with 10 ranks, means all values in (0.2,0.3) are
    // sent to the third rank (with ID 2)
    for (auto i = 0; i < nlocal; i++) {
        // Determine rank each datum will be sent to (based on value of x-coordinate)
        auto new_rank = std::floor(data[i].coord[0] / segment_width);
        data[i].rank = new_rank;
        if (new_rank != world_rank) // Don't increment nsend and total_send when datum stays on same rank
        {
            nsend[new_rank]++;
            total_send++;
        } else // Datum stays on current rank
        {
            data[i].rank = -1;
        }
    }

    // Sort the data by the rank it is going to be sent to
    // Use lambda to sort by `data.rank`
    std::sort(data.begin(), data.end(), [](const Data &a, const Data &b) { return a.rank < b.rank; });
    LOG(world_rank) << " Sending " << total_send << " amount of data, requiring "
                    << static_cast<double>(sizeof(Data) * total_send) / 1024. / 1024. / 1024. << "GB" << std::endl;

    // Make buffer to hold all the data
    // that is to be sent to other ranks
    std::vector<Data> send_buffer(total_send);
    // All entries in `data` before `nlocal - total_send` are staying on their current rank
    for (auto i = nlocal - total_send; i < nlocal; i++) {
        auto idx = i - (nlocal - total_send);
        auto irank = data[i].rank;
        send_buffer[idx] = data[i];
    }

    // Allgather is needed so each rank has a record of how many data
    // it receives from every other rank
    // Allreduce just gives total received by each rank, which isn't
    // enough info to properly perform the send/receives
    auto p1 = nsend.data();
    auto p2 = nrecv.data();
    // MPI_Allreduce(p1, p2, world_size, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgather(p1, world_size, MPI_INTEGER, p2, world_size, MPI_INTEGER, MPI_COMM_WORLD);

    // Sum up the `nrecv` entries for each rank to get `total_recv`
    // nrecv[0:world_size-1] = nsend_0, nrecv[world_size:2*world_size-1] = nsend_1, etc.
    std::vector<int> recv_offsets(world_size);
    for (auto i = 0; i < world_size; i++) {
        total_recv += nrecv[world_rank + (i * world_size)];
    }
    // Make buffer to hold all the data
    // that is to be received from other ranks
    std::vector<Data> recv_buffer(total_recv);
    // Set offsets for recv_buffer for each receive
    recv_offsets[0] = 0;
    int total = 0;
    for (auto i = 1; i < world_size; i++) {
        total += nrecv[world_rank + ((i - 1) * world_size)];
        recv_offsets[i] = total;
    }

    // Use verbosity flag for this
    if (iverbose) {
        LOG(world_rank) << "MPI communication has [total_send, total_recv] = " << total_send << ", " << total_recv
                        << std::endl;
    }

    // Record mem usage after before sending
    MPILog0NodeMemUsage(comm);
    MPILog0NodeSystemMem(comm);
    // Now we need to actually redistribute
    MPI_Barrier(MPI_COMM_WORLD); // Make sure all ranks are synchronised
    // Adjust `data` vector size to updated value after sending/receiving
    auto new_nlocal = nlocal - (total_send - total_recv);
    data.resize(new_nlocal);
    // Tracks index in `send_buffer` where each new rank of data sends begins
    int send_start = 0;
    std::vector<MPI_Request> recvreqs;
    // BLOCKING SEND/RECEIVE
    if (iblocking == 0) {
        for (auto i = 0; i < world_size; i++) {
            if (i != world_rank) {
                int tag = i + (world_rank * world_size);
                auto send_bytes = nsend[i] * sizeof(Data);
                auto recv_rank = i;
                auto send_rank = world_rank;
                void *p1 = &send_buffer[send_start];
                MPI_Send(p1, send_bytes, MPI_BYTE, recv_rank, tag, MPI_COMM_WORLD);
                if (iverbose) {
                    LOG(world_rank) << " Sending from " << send_rank << " to " << recv_rank << std::endl;
                }
                send_start += nsend[i];
            } else {
                nlocal -= total_send; // Adjust nlocal to replace old (sent) value with new (received) ones
                for (auto j = 0; j < world_size; j++) {
                    if (j != world_rank) {
                        int tag = world_rank + (j * world_size);
                        int this_recv = nrecv[i + (j * world_size)];
                        auto recv_rank = world_rank;
                        auto send_rank = j;
                        auto recv_bytes = this_recv * sizeof(Data);
                        void *p2 = &recv_buffer[recv_offsets[j]];
                        MPI_Recv(p2, recv_bytes, MPI_BYTE, send_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (iverbose) {
                            LOG(world_rank) << " Received on " << recv_rank << " from " << send_rank << std::endl;
                        }
                    }
                }
            }
        }
    }
    // NON-BLOCKING SENDS AND RECEIVES
    else {
        // NON-BLOCKING SENDS
        auto nsends = 0, nrecvs = 0;
        std::vector<int> send_ranks;
        for (auto isend = 0; isend < world_size; isend++) {
            if (isend != world_rank) {
                send_ranks.push_back(isend);
            }
        }
        if (irandom) {
            auto rng = std::default_random_engine{};
            std::shuffle(send_ranks.begin(), send_ranks.end(), rng);
        }
        if (idelay) {
            MPI_Barrier(MPI_COMM_WORLD);
            sleep(world_rank * 5);
        }
        for (auto isend : send_ranks) {
            MPI_Request request;
            int tag = isend + (world_rank * world_size);
            auto send_bytes = nsend[isend] * sizeof(Data);
            if (iverbose) {
                LOG(world_rank) << " Sending " << nsend[isend] << " to " << isend << " and has sent a total of "
                                << nsends << " messages" << std::endl;
            }
            if (send_bytes > 0) {
                void *p1 = &send_buffer[send_start];
                MPI_Isend(p1, send_bytes, MPI_BYTE, isend, tag, MPI_COMM_WORLD, &request);
                p1 = nullptr;
                // Add verbosity for timestamp and just stating that MPI_Isend returns (something, anyway)
                send_start += nsend[isend];
            }
            nsends++;
            if (iverbose) {
                LOG(world_rank) << " Sent " << nsend[isend] << " to " << isend << " and has sent a total of " << nsends
                                << " messages" << std::endl;
            }
        }
        if (iverbose) {
            LOG(world_rank) << " Placed " << nsends << " sends " << std::endl;
            MPILogMemUsage();
        }
        // CORRESPONDING NON-BLOCKING RECEIVES
        nlocal -= total_send;
        for (auto j = 0; j < world_size; j++) {
            if (j != world_rank) {
                MPI_Request request;
                int tag = world_rank + (j * world_size);
                int this_recv = nrecv[world_rank + (j * world_size)];
                auto recv_rank = world_rank;
                auto send_rank = j;
                auto recv_bytes = this_recv * sizeof(Data);
                if (recv_bytes > 0) {
                    void *p2 = &recv_buffer[recv_offsets[j]];
                    if (iblocking == 1) {
                        if (iverbose) {
                            LOG(world_rank)
                                << " Receiving " << this_recv << " receives from " << send_rank << std::endl;
                        }
                        MPI_Recv(p2, recv_bytes, MPI_BYTE, send_rank, tag, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                    } else if (iblocking == 2) {
                        if (iverbose) {
                            LOG(world_rank)
                                << " Receiving " << this_recv << " receives from " << send_rank << std::endl;
                        }
                        MPI_Irecv(p2, recv_bytes, MPI_BYTE, send_rank, tag, MPI_COMM_WORLD, &request);
                        MPI_Wait(&request, MPI_STATUSES_IGNORE);
                    } else if (iblocking == 3) {
                        MPI_Irecv(p2, recv_bytes, MPI_BYTE, send_rank, tag, MPI_COMM_WORLD, &request);
                        recvreqs.push_back(request);
                    }
                    p2 = nullptr;
                    nrecvs++;
                }
            }
        }
        if (iblocking == 3) {
            MPI_Waitall(recvreqs.size(), recvreqs.data(), MPI_STATUSES_IGNORE);
            if (iverbose) {
                LOG(world_rank) << " Finished receiving " << nrecvs << " receives " << std::endl;
            }
        }
    }
    // Fill `data`, replacing sent values with received ones
    // adjusting nlocal to new_nlocal in the process
    for (auto i = 0; i < total_recv; i++) {
        data[nlocal++] = recv_buffer[i];
    }

    if (iverbose) {
        LOG(world_rank) << "Now has " << nlocal << " entries" << std::endl;
    }

    // Report time taken
    MPILog0NodeMemUsage(comm);
    MPILog0NodeSystemMem(comm);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    reportTime(start, end, __func__, std::to_string(__LINE__));
}

// TODO: add call to MPI binding
int main(int argc, char *argv[]) {
    // Initial MPI setup
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Report MPI core binding and API (version, etc.)
    auto comm = MPI_COMM_WORLD;
    auto ThisTask = world_rank;
    MPILog0ParallelAPI();
    MPILog0Binding();

    // Ensure program is called with right number of arguments
    if (argc < 2) {
        std::cout << "Usage: pt2pt.out ndata iadjacent iblocking iverbose";
        return 1;
    }

    // Make sure program is only run if there are multiple ranks
    if (world_size == 1) {
        std::cout << "Error: This program must be run with multiple MPI ranks";
        return 1;
    }

    // Parse command line arguments
    unsigned long long ndata = pow(static_cast<unsigned long long>(atoi(argv[1])), 3ull); // Number of data is N^3
    int iadjacent = atoi(argv[2]);
    int iblocking = atoi(argv[3]);
    int iverbose = atoi(argv[4]);
    int idelay = atoi(argv[5]);
    int irandom = atoi(argv[6]);
    // Bounds of distribution used to generate data entries
    float umin = 0.0;
    float umax = 1.0;
    float segment_width = (umax - umin) / world_size;
    // std::cout << "iblocking = " << iblocking << std::endl;
    // Get number of entries for each rank given
    // total data size and no. of ranks
    int nlocal = ndata / world_size;
    // Handle when nprocs doesn't evenly divide into ndata
    // by assigning leftover data to final rank
    if (world_rank == world_size - 1) {
        nlocal = ndata - nlocal * (world_size - 1);
    }
    if (world_rank == 0) {
        LOG(world_rank) << "Generating " << ndata << " in total across comm world and each rank has approximately "
                        << nlocal << std::endl;
    }

    // Report node and code memory usage before any data is generated
    MPILog0NodeMemUsage(comm);
    MPILog0NodeSystemMem(comm);
    // Generate data that will be passed between the ranks
    MPI_Barrier(MPI_COMM_WORLD); // Synchronise tasks
    auto data = generateData(nlocal, umin, umax, segment_width, iadjacent);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronise tasks
    // Report node and code memory usage after data generation, but before reistribution
    MPILog0NodeMemUsage(comm);
    MPILog0NodeSystemMem(comm);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronise tasks

    // Redistribute the data amongst the ranks
    redistributeData(data, nlocal, umax, umin, segment_width, iblocking, iadjacent, iverbose, idelay, irandom);

    // Message for ReFrame test to make sure that job finished
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (world_rank == root_rank) {
        std::cout << "Job completed at " << std::ctime(&end_time) << std::endl;
    }

    MPI_Finalize();
    return 0;
}