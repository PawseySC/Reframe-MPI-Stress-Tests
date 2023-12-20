
#include <logger.h>

void Logger::ReportTimes(std::string name, std::vector<double> &times) 
{
    if (times.size() == 0) return;
    // get average, stddev;
    double ave = 0, stddev = 0, n = static_cast<double>(times.size());
    for (auto &x:times) { ave += x; stddev += x*x;}
    ave /= n;
    stddev = sqrt((stddev - n*ave*ave)/(n-1));
    // sort times and keep track of order
    std::vector<size_t> idx(times.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&times](size_t i1, size_t i2) {return times[i1] < times[i2];});

    std::cout<<"Reporting times for "<<name<<std::endl;
    std::cout<<"Times (min,1,16,median,86,99,max) [us] and corresponding indicies = ";
    std::cout<<"(";
    std::cout<<times[idx[0]]<<", ";
    std::cout<<times[idx[static_cast<int>(n*0.01)]]<<", ";
    std::cout<<times[idx[static_cast<int>(n*0.16)]]<<", ";
    std::cout<<times[idx[static_cast<int>(n*0.50)]]<<", ";
    std::cout<<times[idx[static_cast<int>(n*0.86)]]<<", ";
    std::cout<<times[idx[static_cast<int>(n*0.99)]]<<", ";
    std::cout<<times[idx[static_cast<int>(n-1)]]<<")";
    std::cout<<" : ";
    std::cout<<"(";
    std::cout<<idx[0]<<", ";
    std::cout<<idx[static_cast<int>(n*0.01)]<<", ";
    std::cout<<idx[static_cast<int>(n*0.16)]<<", ";
    std::cout<<idx[static_cast<int>(n*0.50)]<<", ";
    std::cout<<idx[static_cast<int>(n*0.86)]<<", ";
    std::cout<<idx[static_cast<int>(n*0.99)]<<", ";
    std::cout<<idx[static_cast<int>(n-1)]<<")";
    std::cout<<std::endl;
    std::cout<<"Times (ave,stddev) [us] = ";
    std::cout<<"("<<ave<<", "<<stddev<<")"<<std::endl;
}

std::string Logger::ReportGPUSetup(){
    std::string s;
#ifdef _OPENMP 
    s = "OpenMP Target Offloading";
#elif defined(_OPENACC) 
    s = "OpenACC Target Offloading";
#elif defined(USEHIP)
    s = "HIP";
#elif defined(USECUDA)
    s = "CUDA";
#else 
    s = "Unknown";
#endif
    std::cout<<"Code using: "<<s<<std::endl; 
    return s;
} 
