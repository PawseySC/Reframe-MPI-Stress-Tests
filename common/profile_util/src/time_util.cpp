/*! \file time_util.cpp
 *  \brief Get timing
 */

#include "profile_util.h"

/// get the time taken to do some comptue 
namespace profiling_util {
    template <typename T>
    inline
    std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const Timer &t) {
        os << ns_time(t.get());
        return os;
    }

    void profiling_util::StateSampler::_launch()
    {
        std::string cmd;
        std::string s_cpu;
        s_cpu = " ps -p " + std::to_string(pid) + " -o %cpu | tail -n 1";
        
        cmd =  profiling_util::StateSampler::_set_sampling(s_cpu, cpu_usage_fname); 
        (*threads).emplace_back(std::thread(&profiling_util::StateSampler::_place_long_lived_cmd, this, cmd, sample_time));
#ifdef _GPU
        if (use_device) 
        {
            // usage cmd 
            std::vector<std::string> s_gpu_requests = {
                std::string(pu_gpu_usage_request(nDevices)),
                std::string(pu_gpu_energy_request(nDevices)),
                std::string(pu_gpu_mem_request(nDevices)),
                std::string(pu_gpu_memusage_request(nDevices))
            };
            std::vector<std::string> fnames = {
                gpu_usage_fname,
                gpu_energy_fname,
                gpu_mem_fname,
                gpu_memusage_fname
            };
            std::string s_gpu;
            for (auto i=0;i<s_gpu_requests.size();i++) 
            {
                auto req = s_gpu_requests[i];
                auto fname = fnames[i];
                s_gpu = std::string(pu_gpuMonitorCmd) + " " + req + " " + std::string(pu_gpu_formating(nDevices));
                cmd = _set_sampling(s_gpu, fname);
                (*threads).emplace_back(std::thread(&profiling_util::StateSampler::_place_long_lived_cmd, this, cmd, sample_time));
            }
        }
#endif
    }
    profiling_util::StateSampler::StateSampler(const std::string &f, const std::string &l, float _sample_time_in_sec, bool _use_device) : profiling_util::Timer::Timer(f,l,_use_device)
    {
        pid = getpid();
        // set the random seed based on curent time 
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        id = std::rand();
        sample_time = _sample_time_in_sec*1000.0;//convert to micro seconds
        use_device = _use_device;

        cpu_usage_fname = ".sampler.cpu_usage." + std::to_string(id) + ".txt";
#ifdef _GPU
        gpu_usage_fname = ".sampler.gpu_usage."+std::to_string(id)+".txt";
        gpu_energy_fname = ".sampler.gpu_energy."+std::to_string(id)+".txt";
        gpu_mem_fname = ".sampler.gpu_mem."+std::to_string(id)+".txt";
        gpu_memusage_fname = ".sampler.gpu_memusage."+std::to_string(id)+".txt";
        pu_gpuErrorCheck(pu_gpuGetDeviceCount(&nDevices));
#endif
        // allocate the thread vector
        threads = new std::vector<std::thread>;

        _launch();

    }
    profiling_util::StateSampler::~StateSampler()
    {
        Pause();
        delete threads;
        // and remove files
        std::filesystem::remove(cpu_usage_fname);
        std::filesystem::remove(cpu_energy_fname);
#ifdef _GPU
        std::filesystem::remove(gpu_usage_fname);
        std::filesystem::remove(gpu_energy_fname);
        std::filesystem::remove(gpu_mem_fname);
        std::filesystem::remove(gpu_memusage_fname);
#endif

    }
    void profiling_util::StateSampler::Pause()
    {
        std::unique_lock<std::mutex> lock(mtx);
        stopFlag = true;
        cv.notify_all();
        for (auto &t: *threads) t.join();
        (*threads).clear();
    }
    void profiling_util::StateSampler::Restart()
    {
        stopFlag = false;
        cv.notify_all();
        _launch();
    }

    std::vector<double> profiling_util::StateSampler::GetSamplingData(const std::string &fname)
    {
        std::ifstream file(fname);
        std::vector<double> content;
        std::string line;
        while (std::getline(file, line)) {
            double val;
            try {
                val = std::stod(line);
            }
            // on two gcd cards, one will return N/A for power so catch it
            // set val to zero
            catch (const std::invalid_argument& ia) {
                val = 0;
            }
            content.push_back(val);
        }
        file.close();
        return content;
    }


    template <typename T> inline std::string _make_statistics_report(
        const std::string &f, const std::string &l, 
        const std::string ref, profiling_util::detail::_nanoseconds_amount time, 
        const std::string dev, const std::string prop, const std::string unit, 
        T ave, T std, T min, T max
        )
    {
        std::string new_ref = "@"+f+" L"+l;
        std::ostringstream report;
        report <<dev<<" "<<prop<<" ("<<unit<<") statistics taken between : " << new_ref << " - " << ref << " over " << time << " : "; 
        report <<" [ave,std,min,max] = [ "<<ave<<", "<<std<<", "<<min<<", "<<max<<" ] ";
        return report.str();
    }

    std::string ReportCPUUsage(profiling_util::StateSampler &s, 
        const std::string &function, 
        const std::string &line_num)
    {
        s.Pause();
        std::vector<double> content(s.GetSamplingData(s.GetCPUUsageFname()));
        s.Restart();
        auto [ave, std, min, max] = get_stats(content);
        return _make_statistics_report<double>(function, line_num, s.get_ref(), ns_time(s.get()), "CPU", "Usage", "%", ave, std, min, max);
    }
#ifdef _GPU
    std::string ReportGPUUsage(profiling_util::StateSampler &s, 
        const std::string &function, 
        const std::string &line_num, 
        int gid)
    {
        s.Pause();
        std::vector<double> content(std::move(s.GetSamplingData(s.GetGPUEnergyFname())));
        s.Restart();
        auto n = s.GetNumDevices();
        std::ostringstream report;
        for (auto i=0;i<n;i++) 
        {
            auto [ave, std, min, max] = get_stats(content, i, n);
            report <<_make_statistics_report<double>(function, line_num, s.get_ref(), ns_time(s.get()), "GPU"+std::to_string(i), "Usage", "%", ave, std, min, max);
            report <<" | ";
        }
        return report.str();
    }
    std::string ReportGPUEnergy(profiling_util::StateSampler &s, 
        const std::string &function, 
        const std::string &line_num,
        int gid)
    {
        s.Pause();
        std::vector<double> content(std::move(s.GetSamplingData(s.GetGPUEnergyFname())));
        s.Restart();
        auto n = s.GetNumDevices();
        std::ostringstream report;
        for (auto i=0;i<n;i++) 
        {
            auto [ave, std, min, max] = get_stats(content, i, n);
            // to get Wh
            auto energy_used = ave * static_cast<double>(content.size()) * s.GetSampleTime()/1000.0/3600.0;
            report << _make_statistics_report<double>(function, line_num, s.get_ref(), ns_time(s.get()), "GPU"+std::to_string(i), "Power", "W", ave, std, min, max);
            report <<" GPU Energy (Wh) used = "<< energy_used;
            report <<" | ";
        }
        return report.str();
    }
    std::string ReportGPUMem(profiling_util::StateSampler &s, 
        const std::string &function, 
        const std::string &line_num, 
        int gid)
    {
        s.Pause();
        std::vector<double> content(std::move(s.GetSamplingData(s.GetGPUEnergyFname())));
        s.Restart();
        auto n = s.GetNumDevices();
        std::ostringstream report;
        for (auto i=0;i<n;i++) 
        {
            auto [ave, std, min, max] = get_stats(content, i, n);
            report <<_make_statistics_report<double>(function, line_num, s.get_ref(), ns_time(s.get()), "GPU"+std::to_string(i), "Memory", "MiB", ave, std, min, max);
            report <<" | ";
        }
        return report.str();
    }
    std::string ReportGPUMemUsage(profiling_util::StateSampler &s, 
        const std::string &function, 
        const std::string &line_num, 
        int gid)
    {
        s.Pause();
        std::vector<double> content(std::move(s.GetSamplingData(s.GetGPUEnergyFname())));
        s.Restart();
        auto n = s.GetNumDevices();
        std::ostringstream report;
        for (auto i=0;i<n;i++) 
        {
            auto [ave, std, min, max] = get_stats(content, i, n);
            report <<_make_statistics_report<double>(function, line_num, s.get_ref(), ns_time(s.get()), "GPU"+std::to_string(i), "Memory Usage", "%", ave, std, min, max);
            report <<" | ";
        }
        return report.str();
    }


    std::string ReportGPUStatistics(profiling_util::StateSampler &s, 
        const std::string &function, 
        const std::string &line_num, 
        int gid)
    {
        s.Pause();
        auto t = ns_time(s.get());
        auto ref = s.get_ref();
        auto n = s.GetNumDevices();
        std::ostringstream report;
        std::vector<std::string> flist = {s.GetGPUUsageFname(), s.GetGPUMemUsageFname(), s.GetGPUEnergyFname()};
        std::vector<std::string> plist = {"Usage", "Memory Usage", "Power"};
        std::vector<std::string> ulist = {"%", "%", "%"};
        report << "GPU Statistics || ";
        for (auto i=0;i<flist.size();i++) 
        {
            std::vector<double> content(std::move(s.GetSamplingData(flist[i])));
            for (auto j=0;j<n;j++) {
                auto [ave, std, min, max] = get_stats(content, j, n);
                report<< _make_statistics_report<double>(function, line_num, ref, t, "GPU"+std::to_string(j), plist[i], ulist[i], ave, std, min, max);
                if (plist[i] == "Power") 
                {
                    auto energy_used = ave * static_cast<double>(content.size()) * s.GetSampleTime()/1000.0/3600.0;
                    report <<" GPU Energy (Wh) used = "<< energy_used;
                }
                report <<" || ";
            }
        }
        s.Restart();
        return report.str();
    }

#endif

    std::string ReportTimeTaken(
        Timer &t, 
        const std::string &function, 
        const std::string &line_num)
    {
        std::string new_ref = "@"+function+" L"+line_num;
        std::ostringstream report;
        report <<"Time taken between : " << new_ref << " - " << t.get_ref() << " : " << ns_time(t.get());
        return report.str();
    }

    float GetTimeTaken(
        Timer &t, 
        const std::string &function, 
        const std::string &line_num)
    {
        return static_cast<float>((t.get()));
    }

#if defined(_GPU)
    std::string ReportTimeTakenOnDevice(
        Timer &t, 
        const std::string &function, 
        const std::string &line_num)
    {
        std::string new_ref = "@"+function+" L"+line_num;
        std::ostringstream report;
        if (t.get_use_device()) report << "Time taken on device between : " ;
        else report << "NO DEVICE to measure : ";
        auto ns = ns_time(t.get_on_device());
        report << t.get_device_swap_info();
        report << new_ref << " - " << t.get_ref() << " : " << ns;
        return report.str();
    }

    float GetTimeTakenOnDevice(
        Timer &t, 
        const std::string &function, 
        const std::string &line_num)
    {
        return t.get_on_device();
    }
#endif

} 

