#include "profile_util/include/profile_util.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <unistd.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Not recognised when using craype-network-ucx
#include <mpi.h>

// Prepend log statements with the rank being called as well as function and line number of calling statement
#define LOG(ThisTask)                                                                                                  \
    std::cout << _MPI_calling_rank(ThisTask) << "@" << __func__ << " L" << __LINE__ << " (" << getcurtime() << ") : "

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
int world_rank, world_size;
int root_rank = 0;

void sendToRoot(std::vector<double> &data, int &nlocal, int delay_rank, int delay_time) {
    MPI_Barrier(MPI_COMM_WORLD);
    void *p = nullptr;

    // Add a delay to all ranks but one, such that one
    // rank has a long delay between recv and send
    if (world_rank != delay_rank) {
        sleep(delay_time);
    }

    // Root rank receives data from all other ranks
    if (world_rank == root_rank) {
        for (auto irank = 0; irank < world_size; irank++) {
            if (irank != root_rank) {
                LOG(world_rank) << "Receiving " << nlocal << " data from " << irank << std::endl;
                MPI_Recv(&nlocal, 1, MPI_INT, irank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                data.resize(nlocal);
                p = data.data();
                MPI_Recv(p, nlocal, MPI_DOUBLE, irank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                LOG(world_rank) << "Received " << nlocal << " data from " << irank << std::endl;
            }
        }
    }
    // All non-root ranks send their data to the root rank
    else {
        LOG(world_rank) << "Sending " << nlocal << " data to " << root_rank << std::endl;
        nlocal = data.size();
        p = data.data();
        MPI_Send(&nlocal, 1, MPI_INT, root_rank, 0, MPI_COMM_WORLD);
        MPI_Send(p, nlocal, MPI_DOUBLE, root_rank, 0, MPI_COMM_WORLD);
        LOG(world_rank) << "Sent " << nlocal << " data to " << root_rank << std::endl;
    }
    p = nullptr;
}

// Generate the data locally for each rank
std::vector<double> generateData(int nlocal, float umin, float umax, float segment_width) {
    if (world_rank == root_rank) {
        LOG(world_rank) << "Generating data on all ranks" << std::endl;
    }
    // Declare vector of data for each rank
    std::vector<double> data(nlocal);

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
            data[i] = udist(generator);
        }

        if (world_rank == root_rank) {
            LOG(world_rank) << "Has generated data of length " << nlocal << std::endl;
        }
#ifdef _OMP
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);

    return data;
}

// Check the different types of MPI send operations
// send the correct dat and the correct dats is received
void correctSends(int ndata, std::string send_mode) {
    MPI_Barrier(MPI_COMM_WORLD);
    void *p = nullptr;

    // Generate data in a simple way such that
    // it can be compared before and after sends
    int size = ndata;
    std::vector<double> data(size);
    for (auto &datum : data) {
        datum = world_rank;
    }

    // Root rank receives data from all other ranks
    if (world_rank == root_rank) {
        for (auto irank = 0; irank < world_size; irank++) {
            // Record current size to compare to size after send
            auto old_size = size;
            if (irank != root_rank) {
                LOG(world_rank) << "Receiving " << size << " data from " << irank << std::endl;
                // Send size of vector
                MPI_Recv(&ndata, 1, MPI_INT, irank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                LOG(world_rank) << "Received " << size << " data from " << irank << std::endl;
                if (size != old_size) {
                    LOG(world_rank) << "RECEIVED WRONG VALUE FOR `size` FROM " << irank << std::endl;
                    break;
                    // MPI_Abort(MPI_COMM_WORLD)
                }
                data.resize(size);
                p = data.data();
                // Send vector entries
                MPI_Recv(p, size, MPI_DOUBLE, irank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Record an old version of vector to compare to current version
                std::vector<double> old_data(old_size);
                for (auto &datum : old_data) {
                    datum = irank;
                }
                for (auto i = 0; i < old_size; i++) {
                    if (old_data[i] != data[i]) {
                        LOG(world_rank) << "RECEIVED WRONG VALUE AT INDEX " << i << " OF `data` VECTOR FROM " << irank
                                        << std::endl;
                        break;
                        // MPI_Abort
                    }
                }
            }
        }
    }
    // All non-root ranks send data to root rank
    else {
        LOG(world_rank) << "Sending " << size << " data to " << root_rank << std::endl;
        size = data.size();
        p = data.data();
        // Test the different types of MPI sends
        MPI_Request request;
        if (send_mode == "send") {
            MPI_Send(&size, 1, MPI_INT, root_rank, 0, MPI_COMM_WORLD);
            MPI_Send(p, size, MPI_DOUBLE, root_rank, 0, MPI_COMM_WORLD);
        }
        if (send_mode == "isend") {
            MPI_Isend(&size, 1, MPI_INT, root_rank, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(p, size, MPI_DOUBLE, root_rank, 0, MPI_COMM_WORLD, &request);
        }
        if (send_mode == "ssend") {
            MPI_Ssend(&size, 1, MPI_INT, root_rank, 0, MPI_COMM_WORLD);
            MPI_Ssend(p, size, MPI_DOUBLE, root_rank, 0, MPI_COMM_WORLD);
        }
        LOG(world_rank) << "Sent " << size << " data to " << root_rank << std::endl;
    }
    // Clear memory reserved for `data`
    data.clear();
    data.shrink_to_fit();
    p = nullptr;
}

int main(int argc, char *argv[]) {
    // Initial MPI setup
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Ensure program is called with right number of arguments
    // The number of arguments depends on the mode in which
    // the program is called
    std::string mode = argv[2];
    std::transform(mode.begin(), mode.end(), mode.begin(), toupper);
    if (world_rank == root_rank) {
        if (mode == "HANGING") {
            if (argc < 5) {
                std::cout << "Usage: misc_tests.out ndata HANGING delay_rank delay_time" << std::endl;
                return 1;
            }
        } else if (mode == "CORRECT_SENDS") {
            if (argc < 4) {
                std::cout << "Usage: misc_tests.out ndata CORRECT_SENDS send_mode" << std::endl;
                return 1;
            }
        } else {
            std::cout << "Error: Invalid value of `mode` specified (misc_tests.out ndata mode ...)" << std::endl;
            return 1;
        }
    }
    // Ensure program is called with at least 2 MPI ranks
    if (world_size == 1) {
        std::cout << "Error: This program must be run with at least 2 MPI ranks" << std::endl;
        return 1;
    }

    // Run the MPI hanging test
    if (mode == "HANGING") {
        // Number of data on each rank is `ndata=N^3`
        // So total data count is `world_size * ndata = world_size * N^3` where user passes N
        int ndata = pow(atoi(argv[1]), 3);
        int delay_rank = atoi(argv[3]);
        int delay_time = atoi(argv[4]);
        if (world_rank == root_rank) {
            std::cout << "Data size is " << ndata << std::endl;
        }

        // Bounds of distribution used to generate data entries
        float umin = 0.0;
        float umax = 1.0;
        // Width of data region for each rank
        float segment_width = (umax - umin) / world_size;

        // Get number of entries for each rank given
        // total data size and no. of ranks
        int nlocal = ndata / world_size;
        // Handle when world_size doesn't evenly divide into
        // ndata by assigning leftover data to final rank
        if (world_rank == world_size - 1) {
            nlocal = ndata - nlocal * (world_size - 1);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // Generate data
        auto data = generateData(nlocal, umin, umax, segment_width);

        // Run MPI hanging test
        sendToRoot(data, nlocal, delay_rank, delay_time);
        // Clear memory reserved for `data`
        data.clear();
        data.shrink_to_fit();
    }
    // Run correct send/recv test
    else if (mode == "CORRECT_SENDS") {
        // Run correct send/recv test
        int new_ndata = atoi(argv[1]);
        std::string send_mode = argv[3];
        correctSends(new_ndata, send_mode);
    }

    // Message for ReFrame test to make sure that job finished
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (world_rank == root_rank) {
        std::cout << "Job completed at " << std::ctime(&end_time) << std::endl;
    }

    MPI_Finalize();
    return 0;
}