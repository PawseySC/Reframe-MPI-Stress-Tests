#include "profile_util.h"
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <mpi.h>

// Make sure that the directory contains
// only digits, marking it as a process
int is_pid_dir(const struct dirent *entry) {
    const char *p;

    for (p = entry->d_name; *p; p++) {
        if (!isdigit(*p)) {
            return 0;
        }
    }

    return 1;
}

// Checks for file existence
inline bool is_file_exist(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// Need to write my own `get_memory_usage`
// function to get a specific PID
profiling_util::memory_usage get_memory_usage(std::string pid) {
    profiling_util::memory_usage usage;

    // Open `/proc/<PID>/status` file
    std::string fname = "/proc/" + pid + "/status";
    const char *stat_file = fname.c_str();
    std::ifstream f(stat_file);
    if (!f.is_open()) {
        std::cerr << "Couldn't open " << stat_file << " for memory usage reading" << std::endl;
    }

    // Extract data from memory lines
    for (std::string line; std::getline(f, line);) {
        auto start = line.find("VmSize:");
        if (start != std::string::npos) {
            std::istringstream is(line.substr(start + 7));
            is >> usage.vm.current;
            continue;
        }
        start = line.find("VmPeak:");
        if (start != std::string::npos) {
            std::istringstream is(line.substr(start + 7));
            is >> usage.vm.peak;
            continue;
        }
        start = line.find("VmRSS:");
        if (start != std::string::npos) {
            std::istringstream is(line.substr(start + 6));
            is >> usage.rss.current;
            continue;
        }
        start = line.find("VmHWM:");
        if (start != std::string::npos) {
            std::istringstream is(line.substr(start + 6));
            is >> usage.rss.peak;
            continue;
        }
    }

    // Convert values from bytes to kB
    usage.vm.current *= 1024;
    usage.vm.peak *= 1024;
    usage.rss.current *= 1024;
    usage.rss.peak *= 1024;

    return usage;
}

// My own version of `GetMemUsage` from `profile_util`
// to reflect the fact I am specifying the PID
std::tuple<std::string, profiling_util::memory_usage>
output_memory_usage(const std::string &function, const std::string &line_num, std::string pid) {
    auto memory_usage = get_memory_usage(pid);
    std::ostringstream memory_report;
    // May need to be profiling_util::detail::memory_amount
    auto append_memory_stats = [&memory_report](const char *name, const profiling_util::memory_stats &stats) {
        memory_report << name << " current/peak: " << profiling_util::memory_amount(stats.current) << " / "
                      << profiling_util::memory_amount(stats.peak);
    };
    memory_report << "Memory report @ " << function << " L" << line_num << " PID" << pid << ": ";
    append_memory_stats("VM", memory_usage.vm);
    memory_report << "; ";
    append_memory_stats("RSS", memory_usage.rss);
    return std::make_tuple(memory_report.str(), memory_usage);
}

// My own version of `ReportMemUsage` from `profile_util`
// to reflect the fact that I am specifying the PID
std::string report_memory_usage(const std::string &function, const std::string &line_num, std::string pid) {
    std::string report;
    profiling_util::memory_usage usage;
    std::tie(report, usage) = output_memory_usage(function, line_num, pid);
    return report;
}

// Store running PIDs in a vector
// Code adapted from https://stackoverflow.com/questions/63372288/getting-list-of-pids-from-proc-in-linux
std::tuple<std::vector<std::string>, std::vector<std::string>> getPIDs(int nprocs) {
    std::string pid, pid_host;
    std::vector<std::string> pid_vec(nprocs), host_vec(nprocs);
    const char *procs_file = "procs_list.txt";
    std::ifstream f(procs_file);
    if (!f.is_open()) {
        std::cerr << "Couldn't open " << procs_file << " for reading" << std::endl;
        pid_vec[0] = "0";
        return {pid_vec, host_vec};
    }
    int count = 0;
    for (std::string line; std::getline(f, line);) {
        auto start = line.find("is: ");
        auto end = line.find("node ");
        if (start != std::string::npos) {
            std::istringstream is(line.substr(start + 4));
            is >> pid;
            pid_vec[count] = pid;
        }

        if (end != std::string::npos) {
            std::istringstream is(line.substr(end + 5));
            is >> pid_host;
            host_vec[count] = pid_host;
            count += 1;
        }
    }

    // Close proc file
    f.close();

    return std::make_tuple(pid_vec, host_vec);
}

// Program to periodically report memory consumption
// by running processes in an MPI comms program
int main(int argc, char *argv[]) {
    // Initial MPI setup
    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Start timing
    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    std::cout << "Process listing started at " << std::ctime(&start_time) << std::endl;

    // Print hostname for ReFrame to see
    char hostname[1024];
    gethostname(hostname, 1024);
    std::cout << "Running on node " << hostname << std::endl;

    // Get free system memory on node before anything else is running
    auto text = profiling_util::exec_sys_cmd("free | head -n 2 | tail -n 1");
    std::cout << "System memory on node " << hostname << ": " << text << std::endl;

    // To ensure MPI program is running before we try and get running PIDs
    // NOTE: May need to adjust this
    sleep(3);

    // Memory reporting cadence and number of processes in separate MPI code
    float cadence = atof(argv[1]);
    int nprocs = atoi(argv[2]);

    // Get PIDs which are running in the MPI program
    std::vector<std::string> pids;
    std::vector<std::string> hosts;
    tie(pids, hosts) = getPIDs(nprocs);
    // auto [pids, hosts] = getPIDs(nprocs);
    auto size = pids.size();
    // std::cout << hostname << ": Size = " << size << std::endl;
    // std::cout << hostname << ": First PID is " << pids[0] << std::endl;

    // Report memory until the file `done.txt` exists
    // TODO: Find a better way to do this
    auto kill_condition = is_file_exist("done.txt");
    // This condition should be False when MPI comms program ends
    while (not kill_condition) {
        // Report memory for all processes
        for (auto i = 0; i < size; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (hosts[i] == hostname) {
                std::string mem_usage_report = report_memory_usage(__func__, std::to_string(__LINE__), pids[i]);
                std::cout << "PERIODIC : " << hosts[i] << " : " << mem_usage_report << "." << std::endl;
            }
        }
        // Report sytem memory using `free` cmd
        MPI_Barrier(MPI_COMM_WORLD);
        std::string system_mem_report = profiling_util::ReportSystemMem(__func__, std::to_string(__LINE__));
        std::cout << "PERIODIC : " << hostname << " : " << system_mem_report << "." << std::endl;

        // Cadence for memory reporting
        sleep(cadence);

        // File only exists if MPI program has finished running
        // TODO: Find a better way for this program to detect finish of other program
        if (is_file_exist("done.txt")) {
            std::cout << "File created!" << std::endl;
            break;
        }
    }

    // Mark end of script completion
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Process listing completed at " << std::ctime(&end_time) << std::endl;

    return 0;
}