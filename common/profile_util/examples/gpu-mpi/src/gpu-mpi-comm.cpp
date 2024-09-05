#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>
#include <thread>
#include <profile_util.h>

#ifdef USEOPENMP
#include <omp.h>
#endif
#include <mpi.h>

int ThisTask, NProcs;

#define Rank0LocalLogger() if (ThisTask==0) Log()
#define LogMPITest() Rank0LocalLogger()<<" running "<<mpifunc<< " test"<<std::endl;
#define LogMPIBroadcaster() if (ThisTask == itask) Log()<<" running "<<mpifunc<<" broadcasting "<<sendsize<<" GB"<<std::endl;
#define LogMPISender() Log()<<" Running "<<mpifunc<<" sending "<<sendsize<<" GB"<<std::endl;
#define LogMPIReceiver() if (ThisTask == itask) Log()<<" running "<<mpifunc<<std::endl;
#define LogMPIAllComm() Rank0LocalLogger()<<" running "<<mpifunc<<" all "<<sendsize<<" GB"<<std::endl;
#define Rank0ReportMem() if (ThisTask==0) {LogMemUsage();LogSystemMem();}

/// define what type of sends to use 
#define USESEND 0
#define USESSEND 1
#define USEISEND 2

struct Options
{
    /// @brief what types of communication to test
    //@{
    /// cpu to cpu communication
    bool icpu = false;
    /// gpu to gpu communication
    bool igpu = true;
    //@}
    /// root mpi task 
    int roottask = 0;
    /// other mpi task 
    int othertask = 1;
    /// how to send information task 
    int usesend = USEISEND;

    /// max message size in GB
    double maxgb = 1.0;
    /// max message size in number of doubles 
    int msize = 1000;
    /// number of iterations
    int Niter = 1;
};

std::tuple<int,
    std::vector<MPI_Comm> ,
    std::vector<std::string> ,
    std::vector<int>, 
    std::vector<int>, 
    std::vector<int>>
    MPIAllocateComms()
{
    // number of comms is 2, 4, 8, ... till MPI_COMM_WORLD;
    int numcoms = std::floor(log(static_cast<double>(NProcs))/log(2.0))+1;
    int numcomsold = numcoms;
    std::vector<MPI_Comm> mpi_comms(numcoms);
    std::vector<std::string> mpi_comms_name(numcoms);
    std::vector<int> ThisLocalTask(numcoms), NProcsLocal(numcoms), NLocalComms(numcoms);

    for (auto i=0;i<=numcomsold;i++) 
    {
        NLocalComms[i] = NProcs/pow(2,i+1);
        if (NLocalComms[i] < 2) 
        {
            numcoms = i+1;
            break;
        }
        auto ThisLocalCommFlag = ThisTask % NLocalComms[i];
        MPI_Comm_split(MPI_COMM_WORLD, ThisLocalCommFlag, ThisTask, &mpi_comms[i]);
        MPI_Comm_rank(mpi_comms[i], &ThisLocalTask[i]);
        MPI_Comm_size(mpi_comms[i], &NProcsLocal[i]);
        int tasktag = ThisTask;
        MPI_Bcast(&tasktag, 1, MPI_INTEGER, 0, mpi_comms[i]);
        mpi_comms_name[i] = "Tag_" + std::to_string(static_cast<int>(pow(2,i+1)))+"_worldrank_" + std::to_string(tasktag);
    }
    mpi_comms[numcoms-1] = MPI_COMM_WORLD;
    ThisLocalTask[numcoms-1] = ThisTask;
    NProcsLocal[numcoms-1] = NProcs;
    NLocalComms[numcoms-1] = 1;
    mpi_comms_name[numcoms-1] = "Tag_world";
    ThisLocalTask.resize(numcoms);
    NLocalComms.resize(numcoms);
    NProcsLocal.resize(numcoms);
    mpi_comms.resize(numcoms);
    mpi_comms_name.resize(numcoms);

    MPI_Barrier(mpi_comms[numcoms-1]);
    return std::make_tuple(numcoms,
        std::move(mpi_comms),
        std::move(mpi_comms_name), 
        std::move(ThisLocalTask), 
        std::move(NProcsLocal), 
        std::move(NLocalComms)
        );
}

void MPIFreeComms(std::vector<MPI_Comm> &mpi_comms, std::vector<std::string> &mpi_comms_name){
    for (auto i=0;i<mpi_comms.size()-1;i++) {
        Rank0LocalLogger()<<"Freeing "<<mpi_comms_name[i]<<std::endl;
        MPI_Comm_free(&mpi_comms[i]);
    }
}

std::vector<unsigned long long> MPISetSize(double maxgb) 
{
    std::vector<unsigned long long> sizeofsends(4);
    sizeofsends[0] = 1024.0*1024.0*1024.0*maxgb/sizeof(double);
    for (auto i=1;i<sizeofsends.size();i++) sizeofsends[i] = sizeofsends[i-1]/8;
    std::sort(sizeofsends.begin(),sizeofsends.end());
    
    if (ThisTask==0) {for (auto &x: sizeofsends) {Log()<<"Messages of "<<x<<" elements and "<<x*sizeof(double)/1024./1024./1024.<<" GB"<<std::endl;}}
    MPI_Barrier(MPI_COMM_WORLD);
    return sizeofsends;
}

std::vector<float> MPIGatherTimeStats(profiling_util::Timer time1, std::string f, std::string l)
{
    std::vector<float> times(NProcs);
    auto p = times.data();
    auto time_taken = profiling_util::GetTimeTaken(time1, f, l);
    MPI_Gather(&time_taken, 1, MPI_FLOAT, p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return times;
}

std::tuple<float, float, float, float> TimeStats(
    std::vector<float> times
) 
{
    std::sort(times.begin(), times.end());
    auto N = times.size();
    auto ave = 0.0, std = 0.0;
    auto mint = times[0];
    auto maxt = times[N-1];
    for (auto &t:times)
    {
        ave += t;
        std += t*t;
        // mint = std::min(mint,t);
        // maxt = std::max(maxt,t);
    }
    float n = times.size();
    ave /= n;
    if (n>1) std = sqrt((std - ave*ave*n)/(n-1.0));
    else std = 0;
    return std::make_tuple(ave, std, mint, maxt);
}

std::tuple<std::vector<float>, std::vector<float>> 
PercentileStats(
    std::vector<float> data, 
    std::vector<float> percentiles = {0, 0.01, 0.16, 0.50, 0.84, 0.99, 1}
) 
{
    std::sort(data.begin(), data.end());
    auto N = data.size();
    std::vector<float> percentile_values;
    percentile_values.push_back(data[0]);
    for (auto &p:percentiles) if (p>0 && p<1) percentile_values.push_back(data[p*static_cast<float>(N)]);
    percentile_values.push_back(data[N-1]);
    return std::make_tuple(percentiles, percentile_values);
}

void MPIReportTimeStats(std::vector<float> times, 
    std::string commname, std::string message_size, 
    std::string f, std::string l)
{
    auto[ave, std, mint, maxt] = TimeStats(times);
    auto[percentiles, values] = PercentileStats(times);
    std::string tinfo;
    tinfo = "MPI Comm=" + commname + " @" + f + ":L" + l 
        + " - message size=" + message_size 
        + " timing [ave,std,min,max]=[" 
        + std::to_string(ave) + "," + std::to_string(std) + "," 
        + std::to_string(mint) + "," + std::to_string(maxt) + "] (microseconds), percentiles [";
    for (auto &p:percentiles) tinfo+=std::to_string(p)+", ";
    tinfo += "] = [";
    for (auto &p:values) tinfo+=std::to_string(p)+", ";
    tinfo += "], ";
    Rank0LocalLogger()<<tinfo<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

void MPIReportTimeStats(profiling_util::Timer time1, 
    std::string commname, std::string message_size, 
    std::string f, std::string l)
{
    auto times = MPIGatherTimeStats(time1, f, l);
    MPIReportTimeStats(times, commname, message_size, f, l);
}

std::string GetTransferAndBandwidth(
    std::vector<float> &times, 
    std::vector<float> &bw
    ) 
{
    auto[percentile1, transfers] = PercentileStats(times);
    auto[percentile2, bandwidth] = PercentileStats(bw);
    std::string perinfo = " Times (s) [";
    for (auto &p:percentile1) perinfo+=std::to_string(p)+", ";
    perinfo += "] = [";
    for (auto &p:transfers) perinfo+=std::to_string(p)+", ";
    perinfo += "], ";
    perinfo += "Bandwidth (GB/s) [";
    for (auto &p:percentile1) perinfo+=std::to_string(p)+", ";
    perinfo += "] = [";
    for (auto &p:bandwidth) perinfo+=std::to_string(p)+", ";
    perinfo += "]";
    return perinfo;
}

/// \defgroup Host MPI (CPU-to-CPU) Performance 
//@{

/// @brief Test host (CPU) send receive
/// @param opt Options struct containing runtime information
void MPITestCPUSendRecv(Options &opt) 
{
    MPI_Status status;
    auto comm_all = MPI_COMM_WORLD;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> senddata, receivedata;
    
    double * p1 = nullptr, *p2 = nullptr;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    // now send receive 
    mpifunc = "CPU_sendrecv";
    LogMPITest();
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        senddata.resize(sizeofsends[i]);
        receivedata.resize(sizeofsends[i]);
        Rank0ReportMem();
        MPILog0NodeMemUsage();
        MPILog0NodeSystemMem();
        for (auto &d:senddata) d = pow(2.0,ThisTask);
        p1 = senddata.data();
        p2 = receivedata.data();
        auto time1 = NewTimer();
        for (auto j=0;j<mpi_comms.size();j++)
        {
#ifdef TURNOFFMPI
#else
            if (ThisLocalTask[j] == 0) {Log()<<"Communicating using comm "<<mpi_comms_name[j]<<std::endl;}
            std::vector<float> times;
            for (auto iter=0;iter<opt.Niter;iter++) {
                auto time2 = NewTimer();
                std::vector<MPI_Request> sendreqs, recvreqs;
                for (auto isend=0;isend<NProcsLocal[j];isend++) 
                {
                    if (isend != ThisLocalTask[j]) 
                    {
                        MPI_Request request;
                        int tag = isend*NProcsLocal[j]+ThisLocalTask[j];
                        MPI_Isend(p1, sizeofsends[i], MPI_DOUBLE, isend, tag, mpi_comms[j], &request);
                        sendreqs.push_back(request);
                    }
                }
                Log()<<" Placed isends "<<std::endl;
                for (auto irecv=0;irecv<NProcsLocal[j];irecv++) 
                {
                    if (irecv != ThisLocalTask[j]) 
                    {
                        MPI_Request request;
                        int tag = ThisLocalTask[j]*NProcsLocal[j]+irecv;
                        MPI_Irecv(p2, sizeofsends[i], MPI_DOUBLE, irecv, tag, mpi_comms[j], &request);
                        recvreqs.push_back(request);
                    }
                }
                Rank0ReportMem();
                MPILog0NodeMemUsage();
                MPILog0NodeSystemMem();
                MPI_Waitall(sendreqs.size(), sendreqs.data(), MPI_STATUSES_IGNORE);
                MPI_Waitall(recvreqs.size(), recvreqs.data(), MPI_STATUSES_IGNORE);
                Log()<<" Received ireceives "<<std::endl;
                auto times_tmp = MPIGatherTimeStats(time2, __func__, std::to_string(__LINE__));
                times.insert(times.end(), times_tmp.begin(), times_tmp.end());
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPIReportTimeStats(times, mpi_comms_name[j], std::to_string(sizeofsends[i]), __func__, std::to_string(__LINE__));
#endif
        }
        if (ThisTask==0) LogTimeTaken(time1);
    }
    senddata.clear();
    senddata.shrink_to_fit();
    receivedata.clear();
    receivedata.shrink_to_fit();
    Rank0ReportMem();
    MPI_Barrier(MPI_COMM_WORLD);
}

/// @brief Test host (CPU) collective
/// @param opt Options struct containing runtime information
void MPITestCPUAllReduce(Options &opt) 
{
    MPI_Status status;
    auto comm_all = MPI_COMM_WORLD;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> data, allreducesum;
    
    double * p1 = nullptr, *p2 = nullptr;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    // now allreduce 
    mpifunc = "CPU_allreduce";
    LogMPITest();
#ifdef TURNOFFMPI
    data.resize(sizeofsends[sizeofsends.size()-1]);
    allreducesum.resize(sizeofsends[sizeofsends.size()-1]);
    for (auto &d:data) d = pow(2.0,ThisTask);
    sleep(10);
    for (auto &d:allreducesum) d = pow(2.0,ThisTask);
#endif
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        data.resize(sizeofsends[i]);
        allreducesum.resize(sizeofsends[i]);
        Rank0ReportMem();
        MPILog0NodeMemUsage();
        MPILog0NodeSystemMem();
        for (auto &d:data) d = pow(2.0,ThisTask);
        p1 = data.data();
        p2 = allreducesum.data();
        auto time1 = NewTimer();
        for (auto j=0;j<mpi_comms.size();j++) 
        {
#ifdef TURNOFFMPI
#else
            if (ThisLocalTask[j] == 0) {Log()<<"Communicating using comm "<<mpi_comms_name[j]<<std::endl;}
            std::vector<float> times;
            for (auto iter=0;iter<opt.Niter;iter++) {
                auto time2 = NewTimer();
                MPI_Allreduce(p1, p2, sizeofsends[i], MPI_DOUBLE, MPI_SUM, mpi_comms[j]);
                auto times_tmp = MPIGatherTimeStats(time2, __func__, std::to_string(__LINE__));
                times.insert(times.end(), times_tmp.begin(), times_tmp.end());
            }
            Rank0ReportMem();
            MPILog0NodeMemUsage();
            MPILog0NodeSystemMem();
            sleep(2);
            MPI_Barrier(MPI_COMM_WORLD);
            MPIReportTimeStats(times, mpi_comms_name[j], std::to_string(sizeofsends[i]), __func__, std::to_string(__LINE__));
#endif
        }
        if (ThisTask==0) LogTimeTaken(time1);
    }
    data.clear();
    data.shrink_to_fit();
    allreducesum.clear();
    allreducesum.shrink_to_fit();
    MPIFreeComms(mpi_comms, mpi_comms_name);
    Rank0ReportMem();
    MPI_Barrier(MPI_COMM_WORLD);
}

/// @brief Test whether the MPI interface sends the correct data 
/// @param opt Options struct containing runtime information
void MPITestCPUCorrectSendRecv(Options &opt) 
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status status;
    auto comm_all = MPI_COMM_WORLD;
    std::string mpifunc;
    unsigned long size = 5;
    std::vector<double> data(size);
    void * p1 = nullptr, *p2 = nullptr;

    mpifunc = "CPU_correct_values";
    for (auto &d:data) d = pow(2.0,ThisTask);
    auto time1 = NewTimer();
    if (ThisTask == opt.roottask) 
    {
        auto oldsize = size;
        for (auto itask = 0;itask<NProcs;itask++) 
        {
            if (itask == opt.roottask) continue;
            int mpi_err;
            Log()<<" receiving from "<<itask<<std::endl;
            mpi_err = MPI_Recv(&size, 1, MPI_UNSIGNED_LONG, itask, 0, MPI_COMM_WORLD, &status);
            Log()<<" size "<<size<<" received from "<<itask<<" with " <<mpi_err<<std::endl;
            if (size != oldsize) {
                Log()<<" GOT WRONG SIZE VALUE from "<<itask<<std::endl;
                MPI_Abort(MPI_COMM_WORLD,8);
            }
            data.resize(size);
            p1 = data.data();
            mpi_err = MPI_Recv(p1, size, MPI_DOUBLE, itask, 0, MPI_COMM_WORLD, &status);
            std::vector<double> refdata(oldsize);
            for (auto &d:refdata) d = pow(2.0,itask);
            for (auto i=0;i<oldsize;i++) {
                if (data[i] != refdata[i]) {
                    Log()<<" GOT WRONG data VALUE from "<<itask<<std::endl;
                    MPI_Abort(MPI_COMM_WORLD,8);
                }
            }

            std::string s;
            for (auto &d:data) s+=std::to_string(d) + " ";
            Log()<<" received from "<<itask<<" with "<<mpi_err<<std::endl;
        }
    }
    else {
        MPI_Request request;
        int mpi_err;
        Log()<<" sending to "<<opt.roottask<<" with send type of "<<opt.usesend<<std::endl;
        size = data.size();
        p1 = data.data();
        if (opt.usesend == USESEND) {
            mpi_err = MPI_Send(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD);
            mpi_err = MPI_Send(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD);
        }
        else if (opt.usesend == USEISEND) 
        {
            mpi_err = MPI_Isend(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD, &request);
            mpi_err = MPI_Isend(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD, &request);
        }
        else if (opt.usesend == USESSEND) 
        {
            mpi_err = MPI_Ssend(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD);
            mpi_err = MPI_Ssend(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD);
        }
        Log()<<" sent to "<<opt.roottask<<" with "<<mpi_err<<std::endl;
    }
    if (ThisTask==0) LogTimeTaken(time1);
    data.clear();
    data.shrink_to_fit();
    p1 = p2 = nullptr;
}

//@}

/// \defgroup Device MPI (GPU-to-GPU) Performance 
//@{

void MPITestGPUCopy(Options &opt){

    auto comm_all = MPI_COMM_WORLD;
    std::vector<double*> gpu_p1, gpu_p2;
    std::vector<float> memcopytimes, memcopybandwidth;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    int nDevices;
    pu_gpuErrorCheck(pu_gpuGetDeviceCount(&nDevices));
    std::vector<double> senddata;
    gpu_p1.resize(nDevices);
    gpu_p2.resize(nDevices);
    for (auto &x:gpu_p1) x=nullptr;
    for (auto &x:gpu_p2) x=nullptr;

    // now allreduce 
    std::string mpifunc = "GPU_CPU_copy";
    LogMPITest();
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        auto nbytes = sizeofsends[i]*sizeof(double);
        senddata.resize(sizeofsends[i]);
        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<" allocating memory on device "<<idev<<" and transferring "<<nbytes<<std::endl;
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p1[idev], nbytes));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p2[idev], nbytes));

            float timetaken=0.0, bandwidth=0.0;
            pu_gpuEvent_t gpuEventStart, gpuEventStop;
            pu_gpuErrorCheck(pu_gpuEventCreate(&gpuEventStart));
            pu_gpuErrorCheck(pu_gpuEventCreate(&gpuEventStop));
            for (auto iter=0;iter<opt.Niter;iter++) {
                pu_gpuErrorCheck(pu_gpuEventRecord(gpuEventStart));
                pu_gpuErrorCheck(pu_gpuMemcpy(gpu_p1[idev], senddata.data(), nbytes, pu_gpuMemcpyHostToDevice));
                pu_gpuErrorCheck(pu_gpuEventRecord(gpuEventStop));
                pu_gpuErrorCheck(pu_gpuDeviceSynchronize());
                pu_gpuErrorCheck(pu_gpuEventElapsedTime(&timetaken, gpuEventStart, gpuEventStop));
                memcopytimes.push_back(timetaken*_GPU_TO_SECONDS);
                memcopybandwidth.push_back(nbytes/1024.0/1024.0/1024.0/(timetaken*_GPU_TO_SECONDS));
            }
            pu_gpuErrorCheck(pu_gpuEventDestroy(gpuEventStart));
            pu_gpuErrorCheck(pu_gpuEventDestroy(gpuEventStop));
            auto s = GetTransferAndBandwidth(memcopytimes,memcopybandwidth);
            Log()<<" GPU "<<idev<<":"<<s<<std::endl;
            pu_gpuErrorCheck(pu_gpuFree(gpu_p1[idev]));
            pu_gpuErrorCheck(pu_gpuFree(gpu_p2[idev]));
            memcopytimes.clear();
            memcopybandwidth.clear();
        }
    }
}

/// @brief Test whether simple send/receive works and check bandwidth
/// @param opt Options struct containing runtime information
void MPITestGPUBandwidthSendRecv(Options &opt){
    if (NProcs<2) return;
    MPI_Status status;
    auto comm_all = MPI_COMM_WORLD;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> senddata, receivedata;
    
    double * p1 = nullptr, *p2 = nullptr;
    //double *gpu_p1 = nullptr, *gpu_p2 = nullptr;
    std::vector<double*> gpu_p1, gpu_p2;
    std::vector<float> mpitransfertimes, mpitransferbandwidth;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    int nDevices;
    pu_gpuErrorCheck(pu_gpuGetDeviceCount(&nDevices));
    gpu_p1.resize(nDevices);
    gpu_p2.resize(nDevices);
    for (auto &x:gpu_p1) x=nullptr;
    for (auto &x:gpu_p2) x=nullptr;

    // now bandwidth measurement for send receive 
    mpifunc = "GPU_bandwidth_sendrecv";
    LogMPITest();
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        auto nbytes = sizeofsends[i]*sizeof(double);
        senddata.resize(sizeofsends[i]);
        receivedata.resize(sizeofsends[i]);
        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<" allocating memory on device "<<idev<<std::endl;
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p1[idev], nbytes));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p2[idev], nbytes));
            pu_gpuErrorCheck(pu_gpuMemcpy(gpu_p1[idev], senddata.data(), nbytes, pu_gpuMemcpyHostToDevice));
        }
        Rank0ReportMem();
        MPILog0NodeMemUsage();
        MPILog0NodeSystemMem();
        for (auto &d:senddata) d = pow(2.0,ThisTask);
        // p1 = senddata.data();
        // p2 = receivedata.data();
        auto time1 = NewTimer();
        // for (auto j=0;j<mpi_comms.size();j++)
        {
            auto j=mpi_comms.size()-1;

            // need to move this to an async mpi test 
            for (auto idev=0; idev<nDevices;idev++) {
                pu_gpuErrorCheck(pu_gpuSetDevice(idev));
                float timetaken=0.0, bandwidth=0.0;
                pu_gpuEvent_t gpuEventStart, gpuEventStop;
                pu_gpuErrorCheck(pu_gpuEventCreate(&gpuEventStart));
                pu_gpuErrorCheck(pu_gpuEventCreate(&gpuEventStop));
                p1 = gpu_p1[idev];
                p2 = gpu_p2[idev];
                if (ThisLocalTask[j] == 0) {Log()<<"Communicating using comm "<<mpi_comms_name[j]<<" with device "<<idev<<std::endl;}
                std::vector<float> times;
                // need to make pair of sending tasks and receiving tasks
                for (auto itask=0;itask<NProcs;itask++) {
                    if (itask == opt.roottask) continue;
                    if (!(ThisTask==opt.roottask or ThisTask==itask)) continue;
                    int tag = 100;
                    for (auto iter=0;iter<opt.Niter;iter++) {
                        pu_gpuErrorCheck(pu_gpuEventRecord(gpuEventStart));
                        if (ThisTask==opt.roottask) {
                            MPI_Send(p1, sizeofsends[i], MPI_DOUBLE, itask, tag, mpi_comms[j]);
                        }
                        else if (ThisTask==itask) {
                            MPI_Recv(p2, sizeofsends[i], MPI_DOUBLE, opt.roottask, tag, mpi_comms[j], &status);
                        }
                        pu_gpuErrorCheck(pu_gpuEventRecord(gpuEventStop));
                        pu_gpuErrorCheck(pu_gpuDeviceSynchronize());
                        pu_gpuErrorCheck(pu_gpuEventElapsedTime(&timetaken, gpuEventStart, gpuEventStop));
                        mpitransfertimes.push_back(timetaken*_GPU_TO_SECONDS);
                        mpitransferbandwidth.push_back(nbytes/1024.0/1024.0/1024.0/(timetaken*_GPU_TO_SECONDS));
                    }
                    auto s = GetTransferAndBandwidth(mpitransfertimes,mpitransferbandwidth);
                    Log()<<" GPU "<<idev<<" [send,receive] = ["<<opt.roottask<<","<<itask<<"] :"<<s<<std::endl;
                    mpitransfertimes.clear();
                    mpitransferbandwidth.clear();
                }
            }
        }

        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<" Freeing memory on "<<idev<<std::endl;
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuFree(gpu_p1[idev]));
            pu_gpuErrorCheck(pu_gpuFree(gpu_p2[idev]));
        }
    }
    senddata.clear();
    senddata.shrink_to_fit();
    receivedata.clear();
    receivedata.shrink_to_fit();
    gpu_p1.clear();
    gpu_p2.clear();
    Rank0ReportMem();
    MPI_Barrier(MPI_COMM_WORLD);
};

/// @brief Test whether GPU-GPU Asynchronous communication works
/// @param opt Options struct containing runtime information
void MPITestGPUAsyncSendRecv(Options &opt){
    if (NProcs<2) return;
    MPI_Status status;
    auto comm_all = MPI_COMM_WORLD;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> senddata, receivedata;
    
    double * p1 = nullptr, *p2 = nullptr;
    //double *gpu_p1 = nullptr, *gpu_p2 = nullptr;
    std::vector<double*> gpu_p1, gpu_p2;
    std::vector<float> memcopytimes, memcopybandwidth;
    std::vector<float> mpitransfertimes, mpitransferbandwidth;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    int nDevices;
    pu_gpuErrorCheck(pu_gpuGetDeviceCount(&nDevices));
    gpu_p1.resize(nDevices);
    gpu_p2.resize(nDevices);
    for (auto &x:gpu_p1) x=nullptr;
    for (auto &x:gpu_p2) x=nullptr;

    // now allreduce 
    mpifunc = "GPU_async_sendrecv";
    LogMPITest();
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        auto nbytes = sizeofsends[i]*sizeof(double);
        senddata.resize(sizeofsends[i]);
        receivedata.resize(sizeofsends[i]);
        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<" allocating memory on device "<<idev<<std::endl;
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p1[idev], nbytes));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p2[idev], nbytes));
            pu_gpuErrorCheck(pu_gpuMemcpy(gpu_p1[idev], senddata.data(), nbytes, pu_gpuMemcpyHostToDevice));
        }
        Rank0ReportMem();
        MPILog0NodeMemUsage();
        MPILog0NodeSystemMem();
        for (auto &d:senddata) d = pow(2.0,ThisTask);
        // p1 = senddata.data();
        // p2 = receivedata.data();
        auto time1 = NewTimer();
        // for (auto j=0;j<mpi_comms.size();j++)
        {
            auto j=mpi_comms.size()-1;

            // need to move this to an async mpi test 
            for (auto idev=0; idev<nDevices;idev++) {
                pu_gpuErrorCheck(pu_gpuSetDevice(idev));
                p1 = gpu_p1[idev];
                p2 = gpu_p2[idev];
                if (ThisLocalTask[j] == 0) {Log()<<"Communicating using comm "<<mpi_comms_name[j]<<" with device "<<idev<<std::endl;}
                std::vector<float> times;
                for (auto iter=0;iter<opt.Niter;iter++) {
                    auto time2 = NewTimerHostOnly();
                    std::vector<MPI_Request> sendreqs, recvreqs;
                    for (auto isend=0;isend<NProcsLocal[j];isend++) {
                        if (isend != ThisLocalTask[j]) 
                        {
                            MPI_Request request;
                            int tag = isend*NProcsLocal[j]+ThisLocalTask[j]+idev;
                            MPI_Isend(p1, sizeofsends[i], MPI_DOUBLE, isend, tag, mpi_comms[j], &request);
                            sendreqs.push_back(request);
                        }
                    }
                    // Log()<<" Placed isends "<<sendreqs.size()<<std::endl;
                    for (auto irecv=0;irecv<NProcsLocal[j];irecv++) {
                        if (irecv != ThisLocalTask[j]) 
                        {
                            MPI_Request request;
                            int tag = ThisLocalTask[j]*NProcsLocal[j]+irecv+idev;
                            MPI_Irecv(p2, sizeofsends[i], MPI_DOUBLE, irecv, tag, mpi_comms[j], &request);
                            recvreqs.push_back(request);
                        }
                    }
                    // Log()<<" Received ireceives "<<recvreqs.size()<<std::endl;
                    MPI_Waitall(sendreqs.size(), sendreqs.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(recvreqs.size(), recvreqs.data(), MPI_STATUSES_IGNORE);
                    auto times_tmp = MPIGatherTimeStats(time2, __func__, std::to_string(__LINE__));
                    times.insert(times.end(), times_tmp.begin(), times_tmp.end());
                    // Log()<<" finished send recevies"<<std::endl;
                }
                Log()<<" completed send receives "<<std::endl;
                MPI_Barrier(MPI_COMM_WORLD);
                MPIReportTimeStats(times, mpi_comms_name[j], std::to_string(sizeofsends[i]), __func__, std::to_string(__LINE__));
            }
        }
        if (ThisTask==0) LogTimeTaken(time1);

        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<" Freeing memory on "<<idev<<std::endl;
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuFree(gpu_p1[idev]));
            pu_gpuErrorCheck(pu_gpuFree(gpu_p2[idev]));
        }
    }
    senddata.clear();
    senddata.shrink_to_fit();
    receivedata.clear();
    receivedata.shrink_to_fit();
    gpu_p1.clear();
    gpu_p2.clear();
    Rank0ReportMem();
    MPI_Barrier(MPI_COMM_WORLD);
};

/// @brief Check that message sent is correct
/// @param opt Options struct containing runtime information
void MPITestGPUCorrectSendRecv(Options &opt){
    if (NProcs<2) return;
    MPI_Status status;
    auto comm_all = MPI_COMM_WORLD;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> senddata, receivedata;
    
    double * p1 = nullptr, *p2 = nullptr;
    std::vector<double*> gpu_p1, gpu_p2;
    std::vector<float> memcopytimes, memcopybandwidth;
    std::vector<float> mpitransfertimes, mpitransferbandwidth;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    int nDevices;
    pu_gpuErrorCheck(pu_gpuGetDeviceCount(&nDevices));
    gpu_p1.resize(nDevices);
    for (auto &x:gpu_p1) x=nullptr;

    // now allreduce 
    mpifunc = "GPU_correct_sendrecv";
    LogMPITest();
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        auto nbytes = sizeofsends[i]*sizeof(double);
        senddata.resize(sizeofsends[i]);
        receivedata.resize(sizeofsends[i]);
        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<"Allocating memory on device "<<idev<<" with [n,nbytes]=["<<sizeofsends[i]<<","<<nbytes<<"]"<<std::endl;
            for (auto &d:senddata) d = pow(2.0,ThisTask+idev+1);
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p1[idev], nbytes));
            pu_gpuErrorCheck(pu_gpuMemcpy(gpu_p1[idev], senddata.data(), nbytes, pu_gpuMemcpyHostToDevice));
        }
        auto time1 = NewTimer();
       
        // need to move this to an async mpi test 
        for (auto idev=0; idev<nDevices;idev++) {
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            p1 = gpu_p1[idev];
            Rank0LocalLogger()<<"Communicating using device "<<idev<<std::endl;
            if (ThisTask == opt.roottask) {
                for (auto itask = 0;itask<NProcs;itask++) {
                    if (itask == opt.roottask) continue;
                    int mpi_err;
                    Log()<<"Receiving from "<<itask<<std::endl;
                    mpi_err = MPI_Recv(p1, sizeofsends[i], MPI_DOUBLE, itask, 0, MPI_COMM_WORLD, &status);
                    pu_gpuErrorCheck(pu_gpuMemcpy(receivedata.data(), gpu_p1[idev], nbytes, pu_gpuMemcpyDeviceToHost));
                    std::vector<double> refdata(sizeofsends[i]);
                    for (auto &d:refdata) d = pow(2.0,itask+idev+1);
                    for (auto i=0;i<sizeofsends[i];i++) {
                        if (receivedata[i] != refdata[i]) {
                            Log()<<"GOT WRONG data VALUE from "<<itask<<" [index, value, refvalue] = ["<<i<<","<<receivedata[i]<<","<<refdata[i]<<"]"<<std::endl;
                            MPI_Abort(MPI_COMM_WORLD,8);
                        }
                    }
                }
            }
            else {
                MPI_Request request;
                int mpi_err;
                Log()<<"Sending to "<<opt.roottask<<" with send type of "<<opt.usesend<<std::endl;
                if (opt.usesend == USESEND) {
                    mpi_err = MPI_Send(p1, sizeofsends[i], MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD);
                }
                else if (opt.usesend == USEISEND) 
                {
                    mpi_err = MPI_Isend(p1, sizeofsends[i], MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                }
                else if (opt.usesend == USESSEND) 
                {
                    mpi_err = MPI_Ssend(p1, sizeofsends[i], MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD);
                }
                Log()<<"Sent to "<<opt.roottask<<" with "<<mpi_err<<std::endl;

            }
            Log()<<"Finished communication using device "<<idev<<std::endl;
        }

        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<" Freeing memory on "<<idev<<std::endl;
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuFree(gpu_p1[idev]));
        }
    }
    senddata.clear();
    senddata.shrink_to_fit();
    receivedata.clear();
    receivedata.shrink_to_fit();
    gpu_p1.clear();
    gpu_p2.clear();
    Rank0ReportMem();
    MPI_Barrier(MPI_COMM_WORLD);
};

/// @brief Test collective GPU-GPU communication works
/// @param opt Options struct containing runtime information
void MPITestGPUAllReduce(Options &opt){
    if (NProcs<2) return;
    MPI_Status status;
    auto comm_all = MPI_COMM_WORLD;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> senddata, receivedata;
    
    double * p1 = nullptr, *p2 = nullptr;
    //double *gpu_p1 = nullptr, *gpu_p2 = nullptr;
    std::vector<double*> gpu_p1, gpu_p2;
    std::vector<float> mpitransfertimes, mpitransferbandwidth;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    int nDevices;
    pu_gpuErrorCheck(pu_gpuGetDeviceCount(&nDevices));
    gpu_p1.resize(nDevices);
    gpu_p2.resize(nDevices);
    for (auto &x:gpu_p1) x=nullptr;
    for (auto &x:gpu_p2) x=nullptr;

    // now bandwidth measurement for send receive 
    mpifunc = "GPU_allreduce";
    LogMPITest();
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        auto nbytes = sizeofsends[i]*sizeof(double);
        senddata.resize(sizeofsends[i]);
        receivedata.resize(sizeofsends[i]);
        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<" allocating memory on device "<<idev<<std::endl;
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p1[idev], nbytes));
            pu_gpuErrorCheck(pu_gpuHostMalloc((void**)&gpu_p2[idev], nbytes));
            pu_gpuErrorCheck(pu_gpuMemcpy(gpu_p1[idev], senddata.data(), nbytes, pu_gpuMemcpyHostToDevice));
        }
        Rank0ReportMem();
        MPILog0NodeMemUsage();
        MPILog0NodeSystemMem();
        for (auto &d:senddata) d = pow(2.0,ThisTask);
        auto time1 = NewTimer();
        for (auto j=0;j<mpi_comms.size();j++)
        {
            // need to move this to an async mpi test 
            for (auto idev=0; idev<nDevices;idev++) {
                pu_gpuErrorCheck(pu_gpuSetDevice(idev));
                float timetaken=0.0, bandwidth=0.0;
                pu_gpuEvent_t gpuEventStart, gpuEventStop;
                pu_gpuErrorCheck(pu_gpuEventCreate(&gpuEventStart));
                pu_gpuErrorCheck(pu_gpuEventCreate(&gpuEventStop));
                p1 = gpu_p1[idev];
                p2 = gpu_p2[idev];
                if (ThisLocalTask[j] == 0) {Log()<<"Communicating using comm "<<mpi_comms_name[j]<<" with device "<<idev<<std::endl;}
                std::vector<float> times;
                for (auto iter=0;iter<opt.Niter;iter++) {
                    pu_gpuErrorCheck(pu_gpuEventRecord(gpuEventStart));
                    MPI_Allreduce(p1, p2, sizeofsends[i], MPI_DOUBLE, MPI_SUM, mpi_comms[j]);
                    pu_gpuErrorCheck(pu_gpuEventRecord(gpuEventStop));
                    pu_gpuErrorCheck(pu_gpuDeviceSynchronize());
                    pu_gpuErrorCheck(pu_gpuEventElapsedTime(&timetaken, gpuEventStart, gpuEventStop));
                    mpitransfertimes.push_back(timetaken*_GPU_TO_SECONDS);
                    mpitransferbandwidth.push_back(nbytes/1024.0/1024.0/1024.0/(timetaken*_GPU_TO_SECONDS));
                }
                auto s = GetTransferAndBandwidth(mpitransfertimes,mpitransferbandwidth);
                Log()<<" GPU "<<idev<<" collective:"<<s<<std::endl;
                mpitransfertimes.clear();
                mpitransferbandwidth.clear();
            }
        }

        for (auto idev=0;idev<nDevices;idev++) {
            Log()<<" Freeing memory on "<<idev<<std::endl;
            pu_gpuErrorCheck(pu_gpuSetDevice(idev));
            pu_gpuErrorCheck(pu_gpuFree(gpu_p1[idev]));
            pu_gpuErrorCheck(pu_gpuFree(gpu_p2[idev]));
        }
    }
    senddata.clear();
    senddata.shrink_to_fit();
    receivedata.clear();
    receivedata.shrink_to_fit();
    gpu_p1.clear();
    gpu_p2.clear();
    Rank0ReportMem();
    MPI_Barrier(MPI_COMM_WORLD);
};

//@}

void MPIRunTests(Options &opt)
{
    auto comm_all = MPI_COMM_WORLD;
    if (opt.icpu) 
    {
        MPITestCPUSendRecv(opt);
        MPITestCPUCorrectSendRecv(opt);
        MPITestCPUAllReduce(opt);
    }
    if (opt.igpu) 
    {
        MPITestGPUCopy(opt);
        MPITestGPUCorrectSendRecv(opt);
        MPITestGPUBandwidthSendRecv(opt);
        MPITestGPUAsyncSendRecv(opt);
        MPITestGPUAllReduce(opt);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &NProcs);
    MPI_Comm_rank(comm, &ThisTask);
    MPISetLoggingComm(comm);
    Options opt;

    // init logger time
    Rank0LocalLogger()<<"Starting job "<<std::endl;
    Rank0ReportMem();
    MPILog0NodeMemUsage();
    MPILog0NodeSystemMem();
    if (argc >= 2) opt.maxgb = atof(argv[1]);
    if (argc == 3) opt.Niter = atof(argv[2]);
    opt.othertask = NProcs/2 + 1;
    // if (argc >= 2) opt.delay = atoi(argv[1]);
    // if (argc >= 3) opt.msize = atoi(argv[2]);
    // if (argc >= 4) opt.othertask = atoi(argv[3]);

    // default value for 2 node tests assuming that same number of tasks per node 
    // ensures that othertask is testing internode communication
    // alter if want intranode communcation to something like opt.roottask + 1;
    opt.othertask = NProcs/2 + 1;
    
    MPILog0ParallelAPI();
    MPILog0Binding();
    MPI_Barrier(MPI_COMM_WORLD);
    MPIRunTests(opt);

    Rank0LocalLogger()<<"Ending job "<<std::endl;
    MPI_Finalize();
    return 0;
}
