/*! \file profile_util.h
 *  \brief this file contains all function prototypes of the code
 */

#ifndef _PROFILE_UTIL
#define _PROFILE_UTIL

#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <tuple>
#include <ostream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <array>
#include <algorithm>
#include <thread>
#include <condition_variable>
#include <filesystem>

#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>


#ifdef _MPI
#include <mpi.h>
#endif 

#include "profile_util_gpu.h"
#include "profile_util_api.h"

namespace profiling_util {

#ifdef _MPI
    extern MPI_Comm __comm;
    extern int __comm_rank;
#endif

    /// function that returns a string of the time at when it is called. 
    std::string __when();
    /// function that converts the mask of thread affinity to human readable string 
    void cpuset_to_cstr(cpu_set_t *mask, char *str);
    /// reports the parallelAPI 
    /// @return string of MPI comm size and OpenMP version and max threads for given rank
    /// \todo needs to be generalized to report parallel API of code and not library
    std::string ReportParallelAPI();
    /// reports binding of MPI comm world and each ranks thread affinity 
    /// @return string of MPI comm rank and thread core affinity 
    std::string ReportBinding();
    /// reports thread affinity within a given scope, thus depends if called within OMP region 
    /// @param func function where called in code, useful to provide __func__ and __LINE
    /// @param line code line number where called
    /// @return string of thread core affinity 
    std::string ReportThreadAffinity(std::string func, std::string line);
#ifdef _MPI
    /// reports thread affinity within a given scope, thus depends if called within OMP region, MPI aware
    /// @param func function where called in code, useful to provide __func__ and __LINE
    /// @param line code line number where called
    /// @param comm MPI communicator
    /// @return string of MPI comm rank and thread core affinity 
    std::string MPIReportThreadAffinity(std::string func, std::string line, MPI_Comm &comm);
#endif


     /// run a command
    /// @param cmd string of command to run on system
    /// @return string of MPI comm rank and thread core affinity 
    std::string exec_sys_cmd(std::string cmd);

    namespace detail {

        template <int N, typename T>
        struct _fixed {
            T _val;
        };

        template <typename T, int N, typename VT>
        inline
        std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, detail::_fixed<N, VT> v)
        {
            os << std::setprecision(N) << std::fixed << v._val;
            return os;
        }

    } // namespace detail

    ///
    /// Sent to a stream object, this manipulator will print the given value with a
    /// precision of N decimal places.
    ///
    /// @param v The value to send to the stream
    ///
    template <int N, typename T>
    inline
    detail::_fixed<N, T> fixed(T v) {
        return {v};
    }

    namespace detail {

        struct _memory_amount {
            std::size_t _val;
        };

        struct _nanoseconds_amount {
            std::chrono::nanoseconds::rep _val;
        };

        template <typename T>
        inline
        std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const detail::_memory_amount &m)
        {

            if (m._val < 1024) {
                os << m._val << " [B]";
                return os;
            }

            float v = m._val / 1024.;
            const char *suffix = " [KiB]";

            if (v > 1024) {
                v /= 1024;
                suffix = " [MiB]";
            }
            if (v > 1024) {
                v /= 1024;
                suffix = " [GiB]";
            }
            if (v > 1024) {
                v /= 1024;
                suffix = " [TiB]";
            }
            if (v > 1024) {
                v /= 1024;
                suffix = " [PiB]";
            }
            if (v > 1024) {
                v /= 1024;
                suffix = " [EiB]";
            }
            // that should be enough...

            os << fixed<3>(v) << suffix;
            return os;
        }

        template <typename T>
        inline
        std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const detail::_nanoseconds_amount &t)
        {
            auto time = t._val;
	    float ftime = time;
            if (time < 1000) {
                os << time << " [ns]";
                return os;
            }    
	    
	    ftime = time/1000.f;
	    time /= 1000;
	    if (time < 1000) {
                os << time << " [us]";
                return os;
            }

            ftime = time / 1000.f;
            time /= 1000;
            if (time < 1000) {
                os << time << " [ms]";
                return os;
            }

            ftime = time / 1000.f;
            const char *prefix = " [s]";
            if (ftime > 60) {
                ftime /= 60;
                prefix = " [min]";
                if (ftime > 60) {
                    ftime /= 60;
                    prefix = " [h]";
                    if (ftime > 24) {
                        ftime /= 24;
                        prefix = " [d]";
                    }
                }
            }
            // that should be enough...

            os << fixed<3>(ftime) << prefix;
            return os;
        }

    } // namespace detail

    ///
    /// Sent to a stream object, this manipulator will print the given amount of
    /// memory using the correct suffix and 3 decimal places.
    ///
    /// @param v The value to send to the stream
    ///
    inline
    detail::_memory_amount memory_amount(std::size_t amount) {
        return {amount};
    }

    ///
    /// Sent to a stream object, this manipulator will print the given amount of
    /// nanoseconds using the correct suffix and 3 decimal places.
    ///
    /// @param v The value to send to the stream
    ///
    inline
    detail::_nanoseconds_amount ns_time(std::chrono::nanoseconds::rep amount) {
        return {amount};
    }

    struct memory_stats {
        std::size_t current = 0;
        std::size_t peak = 0;
        std::size_t change = 0;
    };

    struct memory_usage {
        memory_stats vm;
        memory_stats rss;
        memory_usage operator+=(const memory_usage& rhs)
        {
            this->vm.current += rhs.vm.current;
            if (this->vm.peak < rhs.vm.peak) this->vm.peak = rhs.vm.peak;
            this->vm.change += rhs.vm.change;

            this->rss.current += rhs.rss.current;
            if (this->rss.peak < rhs.rss.peak) this->vm.peak = rhs.rss.peak;
            this->rss.change += rhs.rss.change;
            return *this;
        };
    };

    struct sys_memory_stats
    {
        std::size_t total;
        std::size_t used;
        std::size_t free;
        std::size_t shared;
        std::size_t cache;
        std::size_t avail;
    };

    ///get memory usage
    memory_usage get_memory_usage();
    ///report memory usage from within a specific function/scope
    ///usage would be from within a function use 
    ///auto l=std::to_string(__LINE__); auto f = __func__; GetMemUsage(f,l);
    std::string ReportMemUsage(const std::string &f, const std::string &l);
    /// like above but also reports change relative to another sampling of memory 
    std::string ReportMemUsage(const memory_usage &prior_mem_use, const std::string &f, const std::string &l);
    /// like ReportMemUsage but also returns the mem usage 
    std::tuple<std::string, memory_usage> GetMemUsage(const std::string &f, const std::string &l);
    std::tuple<std::string, memory_usage> GetMemUsage(const memory_usage &prior_mem_use, const std::string &f, const std::string &l);
    /// Get memory usage on all hosts 
    #ifdef _MPI
    std::string MPIReportNodeMemUsage(MPI_Comm &comm, 
    const std::string &function, 
    const std::string &line_num
    );
    std::tuple<std::string, std::vector<std::string>, std::vector<memory_usage>> MPIGetNodeMemUsage(MPI_Comm &comm, 
    const std::string &function, 
    const std::string &line_num
    );
    #endif


    /// get the memory of the system using free
    sys_memory_stats get_system_memory();
    ///report memory state of the system from within a specific function/scope
    ///usage would be from within a function use 
    ///auto l=std::to_string(__LINE__); auto f = __func__; GetMemUsage(f,l);
    std::string ReportSystemMem(const std::string &f, const std::string &l);
    /// like above but also reports change relative to another sampling of memory 
    std::string ReportSystemMem(const sys_memory_stats &prior_mem_use, const std::string &f, const std::string &l);
    /// like ReportSystemMem but also returns the system memory
    std::tuple<std::string, sys_memory_stats> GetSystemMem(const std::string &f, const std::string &l);
    std::tuple<std::string, sys_memory_stats> GetSystemMem(const sys_memory_stats &prior_mem_use, const std::string &f, const std::string &l);
    #ifdef _MPI
    std::string MPIReportNodeSystemMem(MPI_Comm &comm, const std::string &function, const std::string &line_num);
    std::tuple<std::string, std::vector<std::string>, std::vector<sys_memory_stats>> MPIGetNodeSystemMem(MPI_Comm &comm, const std::string &function, const std::string &line_num);
    #endif

    /// Timer class. 
    /// In code create an instance of time and then just a mantter of 
    /// creating an instance and then reporting it. 
    class Timer {

    public:

        using clock = std::chrono::high_resolution_clock;
        using duration = typename std::chrono::nanoseconds::rep;
        

        /*!
         * Returns whether timer has timer on device and not just host
         *
         * @return boolean on whether timer on device [us]
         */
        inline
        bool get_use_device() const {return use_device;}
        /*!
         * Returns the number of milliseconds elapsed since the reference time
         * of the timer
         *
         * @return The time elapsed since the creation of the timer, in [us]
         */
        inline
        duration get() const {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - tref).count();
        }

        /*!
         * Returns the number of milliseconds elapsed since the creation
         * of the timer
         *
         * @return The time elapsed since the creation of the timer, in [us]
         */
        inline
        duration get_creation() const {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count();
        }

        /*!
         * Returns the elapsed time on device since the reference time
         * of the device event
         *
         * @return The time elapsed since the creation of the timer, in [ns]
         */
#if defined(_GPU)
        inline void get_ref_device() {
            pu_gpuErrorCheck(pu_gpuGetDevice(&other_device_id));
            swap_device = (other_device_id != device_id);
            if (swap_device) {
                pu_gpuErrorCheck(pu_gpuSetDevice(device_id));
            }
        }
        inline void set_cur_device()  {
            if (swap_device) {
                pu_gpuErrorCheck(pu_gpuSetDevice(other_device_id));
            }
        }
        inline
        float get_on_device()  
        {
            if (!use_device) return 0;
            float telapsed = 0;
            get_ref_device();
            pu_gpuEvent_t t1_event;
            // create event 
            pu_gpuErrorCheck(pu_gpuEventCreate(&t1_event));
            pu_gpuErrorCheck(pu_gpuEventRecord(t1_event)); 
            pu_gpuErrorCheck(pu_gpuEventSynchronize(t1_event));
            pu_gpuErrorCheck(pu_gpuEventElapsedTime(&telapsed,t0_event,t1_event));
            telapsed *= _GPU_TO_SECONDS * 1e9; // to convert to nano seconds 
            pu_gpuErrorCheck(pu_gpuEventDestroy(t1_event));
            set_cur_device();
            return telapsed;
        }
#endif

        void set_ref(const std::string &new_ref)
        {
            ref = new_ref;
            t0 = clock::now();
#if defined(_GPU)
            if (use_device) {
                // clean up current event 
                pu_gpuErrorCheck(pu_gpuSetDevice(device_id));
                pu_gpuErrorCheck(pu_gpuEventDestroy(t0_event));
                // make new event on current device
                pu_gpuErrorCheck(pu_gpuGetDevice(&device_id));
                pu_gpuErrorCheck(pu_gpuEventCreate(&t0_event));
                pu_gpuErrorCheck(pu_gpuEventRecord(t0_event)); 
                pu_gpuErrorCheck(pu_gpuEventSynchronize(t0_event));
                other_device_id = device_id;
                swap_device = false;
            }
#endif
        };
        std::string get_ref() const 
        {
            return ref;
        };
#if defined(_GPU)
        std::string get_device_swap_info()
        {
            if (swap_device) {
                return "WARNING: Device swapped during timing: currently on "+ std::to_string(other_device_id) + " but measuring on " + std::to_string(device_id) + " : ";
            }
            else return "";
        };
#endif

        Timer(const std::string &f, const std::string &l, bool _use_device=true) {
            ref="@"+f+" L"+l;
            t0 = clock::now();
            tref = t0;
            use_device = _use_device;
#if defined(_GPU)
            int ndevices;
            pu_gpuErrorCheck(pu_gpuGetDeviceCount(&ndevices));
            if (ndevices == 0) use_device = false;
            if (use_device) {
                pu_gpuErrorCheck(pu_gpuGetDevice(&device_id));
                pu_gpuErrorCheck(pu_gpuEventCreate(&t0_event));
                pu_gpuErrorCheck(pu_gpuEventRecord(t0_event)); 
                pu_gpuErrorCheck(pu_gpuEventSynchronize(t0_event));
                other_device_id = device_id;
            }
#endif
        }
#if defined(_GPU)
        ~Timer()
        {
            if (use_device) {
                if (swap_device) {
                    pu_gpuErrorCheck(pu_gpuSetDevice(device_id));
                    pu_gpuErrorCheck(pu_gpuEventDestroy(t0_event));
                    pu_gpuErrorCheck(pu_gpuSetDevice(other_device_id));
                }
                else {
                    pu_gpuErrorCheck(pu_gpuEventDestroy(t0_event));
                }
            }
        }
#endif

    protected:
        clock::time_point t0;
        clock::time_point tref;
        std::string ref;
        bool use_device = true;
#if defined(_GPU)
        pu_gpuEvent_t t0_event;
        // store the device on which event is recorded
        // and whether timer called on another device
        int device_id, other_device_id;
        bool swap_device = false;
#endif
    };

    /// @brief report the time taken between some reference time (which defaults to creation of timer )
    /// and current call
    /// @param t instance of timer class 
    /// @param f string of function where the ReporTimeTaken is called (at least that is the idea)
    /// @param l string of line number in file where the ReporTimeTaken is called (at least that is the idea)
    /// @return string reporting time taken 
    std::string ReportTimeTaken(Timer &t, const std::string &f, const std::string &l);

    /// @brief get the time taken between some reference time (which defaults to creation of timer )
    /// and current call
    /// @param t instance of timer class 
    /// @param f string of function where the ReporTimeTaken is called (at least that is the idea)
    /// @param l string of line number in file where the ReporTimeTaken is called (at least that is the idea)
    /// @return time taken 
    float GetTimeTaken(Timer &t, const std::string &f, const std::string &l);

#if defined(_GPU)
    /// @brief report the time taken between some reference time (which defaults to creation of timer )
    /// and current call on the device 
    /// @param t instance of timer class 
    /// @param f string of function where the ReporTimeTaken is called (at least that is the idea)
    /// @param l string of line number in file where the ReporTimeTaken is called (at least that is the idea)
    /// @return string reporting time taken 
    std::string ReportTimeTakenOnDevice(Timer &t, const std::string &f, const std::string &l);
    /// @brief get the time taken between some reference time (which defaults to creation of timer )
    /// and current call on device 
    /// @param t instance of timer class 
    /// @param f string of function where the ReporTimeTaken is called (at least that is the idea)
    /// @param l string of line number in file where the ReporTimeTaken is called (at least that is the idea)
    /// @return time taken 
    float GetTimeTakenOnDevice(Timer &t, const std::string &f, const std::string &l);
#endif

    /// @brief get the ave, std, min, max of input vector
    /// @param input input vector
    template <typename T> std::tuple<T,T,T,T>get_stats(std::vector<T> &input, unsigned int offset = 0, unsigned int stride = 1)
    {
        T ave = 0, std = 0, min = 0, max = 0;
        if (input.size()>0) {
            min = max = input[offset];
            for (auto i=offset;i<input.size();i+=stride)
            {
                ave += input[i];
                std += input[i]*input[i];
                min = std::min(input[i], min);
                max = std::max(input[i], max);
            }
            auto n = static_cast<T>(input.size());
            ave /= n;
            if (n == 1) std = 0;
            else std = sqrt(std-ave*ave*n)/(n-1.0);
        }
        return std::tie(ave, std, min, max);
    }

    /// @brief StateSample class that gets the stats of utilisation/energy
    /// from point of creation to requested reporting.
    /// inherents public routines from Timer
    class StateSampler: public profiling_util::Timer {

    private:
        // unique sample identifier
        int id; 
        /// process id
        int pid = 0;
        // time in seconds between samples
        float sample_time = 1.0;
        std::string cpu_energy_fname, cpu_usage_fname, cpu_freq_fname;
        std::vector<std::thread>* threads = nullptr;
        std::mutex mtx;
        std::condition_variable cv;
        bool stopFlag = false;
        bool use_device = true;
        int nDevices = 0;
#ifdef _GPU
        std::string gpu_energy_fname, gpu_usage_fname, gpu_mem_fname, gpu_memusage_fname;
#endif
        
        std::string _set_sampling(const std::string &cmd, const std::string &out)
        {
            // return std::string("while true; do " + cmd + " >> " + out +
            // "; sleep " + std::to_string(sample_time/1000.0) + "; done;");
            return std::string(cmd + " >> " + out);
        }
        /// @brief launches the sampling processes
        void _launch();

        /// @brief Place a command using std::system and threads 
        /// @param cmd command to place 
        void _place_cmd(const std::string cmd)
        {
            auto status = std::system(cmd.c_str());
        }

        /// @brief Place a command using std::system and threads 
        /// @param cmd command to place 
        /// @param sleep_time time to sleep between running command
        void _place_long_lived_cmd(const std::string cmd, float sleep_time)
        {
            while (!stopFlag) 
            {
                auto status = std::system(cmd.c_str());
                usleep(sleep_time);
            }
        }
    public:
        StateSampler(const std::string &f, const std::string &l, float samples_per_sec = 1.0, bool _use_device=true);
        ~StateSampler();
        /// @brief pauses the sampling by joining threads
        void Pause();
        /// @brief restart the sampling by launching threads
        void Restart();
        /// @brief get sample time 
        /// @return sample time
        double GetSampleTime(){return sample_time;}
        /// @brief get file name store cpu usage info
        /// @return filename
        std::string GetCPUUsageFname(){return cpu_usage_fname;}
        /// @brief get file name store cpu energy info
        /// @return filename
        std::string GetCPUEnergyFname(){return cpu_energy_fname;}
#ifdef _GPU
        /// @brief get file name store gpu usage info
        /// @return filename
        std::string GetGPUUsageFname(){return gpu_usage_fname;}
        /// @brief get file name store gpu energy info
        /// @return filename
        std::string GetGPUEnergyFname(){return gpu_energy_fname;}
        /// @brief get file name store gpu mem usage info
        /// @return filename
        std::string GetGPUMemUsageFname(){return gpu_memusage_fname;}
        /// @brief get file name store gpu mem used info
        /// @return filename
        std::string GetGPUMemFname(){return gpu_mem_fname;}
#endif
        /// @brief read the data from a file and returnt the vector of sampling data
        /// @param fname the string of the file name to open
        /// @return vector of data
        std::vector<double> GetSamplingData(const std::string &fname);

        /// @brief return number of devices visible to sampler
        /// @return int of number of devices
        int GetNumDevices(){return nDevices;};
    };

    /// @brief reports the statistics of CPU from start to current line
    /// @param s sampler to use for reporting 
    /// @param f function where called in code, useful to provide __func__ 
    /// @param l code line number where called
    /// @return string of CPU usage statistics
    std::string ReportCPUUsage(StateSampler &s, const std::string &f, const std::string &l);

#ifdef _GPU
    /// @brief reports the statistics of GPU usage from start to current line
    /// @param s sampler to use for reporting 
    /// @param f function where called in code, useful to provide __func__ 
    /// @param l code line number where called
    /// @return string of GPU usage statistics
    std::string ReportGPUUsage(StateSampler &s, const std::string &f, const std::string &l, int gpu_id = -1);

    /// @brief reports the statistics of GPU energy from start to current line
    /// @param s sampler to use for reporting 
    /// @param f function where called in code, useful to provide __func__ 
    /// @param l code line number where called
    /// @return string of GPU energy statistics
    std::string ReportGPUEnergy(StateSampler &s, const std::string &f, const std::string &l, int gpu_id = -1);

    /// @brief reports the statistics of GPU memory used in MiB from start to current line
    /// @param s sampler to use for reporting 
    /// @param f function where called in code, useful to provide __func__ 
    /// @param l code line number where called
    /// @return string of GPU memory used in MiB statistics
    std::string ReportGPUMem(StateSampler &s, const std::string &f, const std::string &l, int gpu_id = -1);

    /// @brief reports the statistics of GPU memory used in % from start to current line
    /// @param s sampler to use for reporting 
    /// @param f function where called in code, useful to provide __func__ 
    /// @param l code line number where called
    /// @return string of GPU memory usage statistics
    std::string ReportGPUMemUsage(StateSampler &s, const std::string &f, const std::string &l, int gpu_id = -1);

    /// reports the GPU statistics 
    /// @param func function where called in code, useful to provide __func__ and __LINE
    /// @param line code line number where called
    /// @param gpu_id gpu device of interest. Default is -1 and gets all gpus
    /// @return string of GPU energy, usage, etc
    std::string ReportGPUStatistics(StateSampler &s, const std::string &f, const std::string &l, int gpu_id = -1);

#ifdef _MPI
    /// MPI wrapper for reporting GPU status
    /// @param func function where called in code, useful to provide __func__ and __LINE
    /// @param line code line number where called
    /// @param comm MPI communicator
    /// @param gpu_id gpu device of interest. Default is -1 and gets all gpus
    /// @return string of GPU energy, usage, etc
    std::string MPIReportGPUStatus(std::string func, std::string line, MPI_Comm &comm, int gpu_id = -1);
#endif
#endif
}

/// \defgroup C_naming
/// Extern C interface
//@{
extern "C" {
    /// \defgropu LogAffinity_C
    //@{
    int report_parallel_api(char *str);
    #define log_parallel_api() printf("@%s L%d %s\n", __func__, __LINE__, profiling_util::ReportParallelAPI().c_str());
    int report_binding(char *str);
    #define log_binding() printf("@%s L%d %s\n", __func__, __LINE__, profiling_util::ReportBinding().c_str());
    int report_thread_affinity(char *str, char *f, int l);
    #define log_thread_affinity() printf("%s\n", profiling_util::ReportThreadAffinity(__func__, std::to_string(__LINE__)).c_str());
    #ifdef _MPI
        #define mpi_log0_thread_affinity() if(ThisTask == 0) printf("%s\n", ReportThreadAffinity(__func__, std::to_string(__LINE__)).c_str());
        #define mpi_log_thread_affinity() printf("%s\n", profiling_util::MPIReportThreadAffinity(__func__, std::to_string(__LINE__)).c_str());
        #define mpi_log0_parallel_api() if(ThisTask==0) printf("@%s L%d %s\n", __func__, __LINE__, profiling_util::ReportParallelAPI().c_str());
        #define mpi_log0_binding() if (ThisTask == 0) printf("@%s L%d %s\n", __func__, __LINE__, profiling_util::ReportBinding().c_str());
    #endif
    //@}

    /// \defgroup LogMem_C
    /// Log memory usage either to std or an ostream
    //@{
    #define log_mem_usage() printf("%s \n", profiling_util::ReportMemUsage(__func__, std::to_string(__LINE__)).c_str());
    #ifdef _MPI
    #define mpi_log_mem_usage() printf("%s %s \n", _MPI_calling_rank.c_str(), profiling_util::ReportMemUsage(__func__, std::to_string(__LINE__)).c_str());
    #endif
    //@}

    /// \defgroup LogTime_C
    /// \todo Still implementing a C friendly structure to record timing. 
    //@{
    struct timer_c {
        double t0, tref;
        char ref[2000], where[2000];
        timer_c(){
            t0 = tref = 0;
            memset(ref, 0, sizeof(ref));
            memset(where, 0, sizeof(where));
        }
        void set_timer_ref(double _t0, char _ref[2000]) {
            t0 = _t0;
            strcpy(ref,_ref);
        }
    };
    void report_time_taken(char *str, timer_c &t);

    //@}
}
//@}

#endif
