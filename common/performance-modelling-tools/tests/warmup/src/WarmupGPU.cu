#include <WarmupGPU.h>
#include <iostream>
#include <cmath>

#if defined(USEHIP) || defined(USECUDA)
/// standard scalar a * vector x plus vector y 
__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}


/// just a vector add to new vector
__global__
void vector_add(float *out, float *a, float *b, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < n) out[id] = a[id] + b[id];
}


/// silly kernel that should be very quick
__global__
void silly_kernel(float *a)
{
    for (int i = 0; i < 2; i++) {
        a[i] + 2*a[i];
    }
}
#endif

void launch_warmup_kernel(int itype, int i, int j, unsigned long long N) 
{
    std::string kernel_type, s;
    float t1;
    float *a, *b;
    if (itype == GPU_ONLY_KERNEL_LAUNCH)
    {
        kernel_type = "KernelLaunchOnly";
        s = kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j); 
        auto mytimer = NewTimer();
#ifdef _OPENMP 
        #pragma omp target
        {
            #pragma omp parallel for
            for (int i = 0; i < 2; i++) {a[i] + 2*a[i];}
        }
        
        // N = 200000000;
        // std::cout<<"memory "<<N*sizeof(float)/1024./1024./1024.<<std::endl;
        // a = new float[N];
        // // #pragma omp target map(tofrom:a[:N])
        // #pragma omp target map(alloc:a[:N])
        // {
        //     #pragma omp parallel for
        //     for (int i = 0; i < N; i++) {
        //         a[i] = 1.0;
        //         a[i] = a[i] + 2*a[i];
        //         a[i] = exp(-sqrt(a[i]));
        //     }
        // }
        std::cout<<s<<" ";
        LogTimeTaken(mytimer);
#elif defined(_OPENACC)
        #pragma acc parallel loop
            for (int i = 0; i < 2; i++) {a[i] + 2*a[i];}
        std::cout<<s<<" ";
        LogTimeTaken(mytimer);
#elif defined(USEHIP) || defined(USECUDA)
        silly_kernel<<<1,1>>>(a);
        LogGPUElapsedTime(s, mytimer);
#endif
    }
    if (itype == GPU_ONLY_MEM_ALLOCATE) 
    {
        kernel_type = "MemAllocOnly";
        s = kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j); 
        auto mytimer = NewTimer();
#ifdef _OPENMP
        // auto a_d = omp_target_alloc(N*sizeof(float), i);
        // omp_target_free(a_d, i);
        #pragma omp target data map(alloc:a[:N])
        {
        }
        std::cout<<s<<" ";
        LogTimeTaken(mytimer);
#elif _OPENACC
        #pragma acc kernels create(a[:N])
        {
        }
        std::cout<<s<<" ";
        LogTimeTaken(mytimer);
#elif defined(USEHIP) || defined(USECUDA)
        gpuMalloc(&a, N*sizeof(float)); 
        gpuFree(a);
        LogGPUElapsedTime(s, mytimer);
#endif
    }
    else if (itype == GPU_ONLY_MEM_TH2D) 
    {
        kernel_type = "tH2D";
        s = kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j); 
        a = new float[N];
#ifdef _OPENMP
        auto mytimer = NewTimer();
        #pragma omp target data map(to:a[:N])
        {
            
        }
        std::cout<<s<<" ";
        LogTimeTaken(mytimer);
#elif defined(_OPENACC)
        auto mytimer = NewTimer();
        #pragma acc kernels copyin(a[:N])
        {
            
        }
        std::cout<<s<<" ";
        LogTimeTaken(mytimer);
#elif defined(USEHIP) || defined(USECUDA)
        gpuMalloc(&b, N*sizeof(float)); 
        auto mytimer = NewTimer();
        gpuMemcpy(b, a, N*sizeof(float), gpuMemcpyHostToDevice);
        LogGPUElapsedTime(s, mytimer);
        gpuFree(b);
#endif
        delete[] a;
    }
    // transfer from device to host
    else if (itype == GPU_ONLY_MEM_TD2H) 
    {
        kernel_type = "tD2H";
        s = kernel_type + " on device " + std::to_string(i) + " round " + std::to_string(j); 
        a = new float[N];
#ifdef _OPENMP
        auto mytimer = NewTimer();
        #pragma omp target data map(from:a[:N])
        {

        }
        std::cout<<s<<" ";
        LogTimeTaken(mytimer);
#elif defined(_OPENACC)
        auto mytimer = NewTimer();
        #pragma acc kernels copyout(a[:N])
        {
            
        }
        std::cout<<s<<" ";
        LogTimeTaken(mytimer);
#elif defined(USEHIP) || defined(USECUDA)
        gpuMalloc(&b, N*sizeof(float)); 
        auto mytimer = NewTimer();
        gpuMemcpy(a, b, N*sizeof(float), gpuMemcpyDeviceToHost);
        LogGPUElapsedTime(s, mytimer);
        gpuFree(b);
#endif
        delete[] a;
    }
}

inline int GetNumDevices()
{
    int deviceCount = 0;
#ifdef _OPENMP 
    deviceCount = omp_get_num_devices();
#elif defined(_OPENACC)
    auto dtype = acc_get_device_type();
    deviceCount = acc_get_num_devices(dtype);
#elif defined(USEHIP) || defined(USECUDA)
    gpuGetDeviceCount(&deviceCount);
#endif
    return deviceCount;
}

inline void SetDevice(int i)
{
#ifdef _OPENMP 
    omp_set_default_device(i);
#elif defined(_OPENACC)
    auto dtype = acc_get_device_type();
    acc_set_device_num(i,dtype);
#elif defined(USEHIP) || defined(USECUDA)
    gpuSetDevice(i);
#endif
}

void warmup_kernel_over_kernels(int rounds, 
    std::vector<int> kernel_order,
    unsigned long long N
)
{
    int deviceCount = GetNumDevices();
    std::cout<<__func__<<" running "<<std::endl;
    auto timeWarmup = NewTimer();
    for (auto i=0;i<deviceCount;i++) 
    {
        // set the device 
        SetDevice(i);
        for (auto j=0;j<rounds;j++) 
        {
            for (auto itype:kernel_order) 
            {
                launch_warmup_kernel(itype, i, j, N);
            }
        }
    }
    LogTimeTaken(timeWarmup);
}

void warmup_kernel_over_rounds(int rounds, int sleeptime, unsigned long long N)
{
    int deviceCount = GetNumDevices();
    std::cout<<__func__<<" running "<<std::endl;
    auto timeWarmup = NewTimer();
    for (auto itype=0;itype<GPU_ONLY_NUM_LAUNCH_TYPES;itype++) 
    {
        for (auto i=0;i<deviceCount;i++) 
        {
            // set the device 
            SetDevice(i);
            for (auto j=0;j<rounds;j++) 
            {
                launch_warmup_kernel(itype, i, j, N);
                if (sleeptime > 0) sleep(sleeptime);
            }
        }
    }
    LogTimeTaken(timeWarmup);
}

void run_on_devices(Logger &logger, int Niter)
{
    int deviceCount = GetNumDevices();
    for (auto i=0;i<deviceCount;i++) 
    {
        SetDevice(i);
        // now check the kernel launches
        std::vector<double> times;
        std::map<std::string, std::vector<double>> device_times;
        std::vector<double> x;
#ifdef _OPENMP 
        device_times.insert({"omp_target",x});
#elif defined(_OPENACC)
        device_times.insert({"acc_target",x});
#elif defined(USEHIP) || defined(USECUDA) 
        device_times.insert({"allocation",x});
        device_times.insert({"tH2D",x});
        device_times.insert({"tD2H",x});
        device_times.insert({"free",x});
        device_times.insert({"kernel",x});
#endif
        for (auto j=0;j<Niter;j++) 
        {
            auto t = NewTimer();
            auto timings = run_kernel(j);
            times.push_back(t.get());
            for (auto &t:timings) 
            {
                device_times[t.first].push_back(t.second);
            }
        }
        std::cout<<"================================="<<std::endl;
        std::cout<<" DEVICE "<<i<<std::endl;
        logger.ReportTimes("run_kernel", times);
        std::cout<<"---------------------------------"<<std::endl;
        std::cout<<"On device times within run_kernel"<<std::endl;
        for (auto &t:device_times) logger.ReportTimes(t.first,t.second);
        std::cout<<"---------------------------------"<<std::endl;
       
    }
}

std::map<std::string, double> run_kernel(int offset)
{
#define gettelapsed(t1)  telapsed = GetTimeTakenOnDevice(t1,__func__, std::to_string(__LINE__));

    std::map<std::string, double> timings;
    float telapsed;
    auto N = 1024*1024;
    float *x, *y, *d_x, *d_y, *out, *d_out;
    x = new float[N];
    y = new float[N];
    out = new float[N];
    for (int i = 0; i < N; i++) 
    {
        x[i] = 1.0*offset;
        y[i] = 2.0*offset;
    }
#ifdef _OPENMP 
    auto tall = NewTimer();
    // auto talloc = NewTimer();
    // auto th2d = NewTimer();
    // auto tk = NewTimer();
    // auto td2h = NewTimer();
    // auto tfree = NewTime();
    #pragma omp target data map(to:x[:N],y[:N]) map(from:out[:N]) 
    {
        #pragma omp target
        #pragma omp parallel for
        for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    }
    // timings.insert({std::string("allocation"), telapsed});
    // timings.insert({std::string("tH2D"), telapsed});
    // timings.insert({std::string("kernel"), telapsed});
    // timings.insert({std::string("tD2H"), telapsed});
    // timings.insert({std::string("free"), telapsed});
    // std::cout<<out[2]<<std::endl;
    // LogTimeTaken(tall);
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("omp_target"), telapsed});
#elif defined(_OPENACC)
    auto tall = NewTimer();
    #pragma acc parallel loop copyin(x[:N],y[:N]) copyout(out[:N])
    for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("acc_target"), telapsed});
#elif defined(USEHIP) || defined(USECUDA)
    auto talloc = NewTimer();
    gpuMalloc(&d_x, N*sizeof(float)); 
    gpuMalloc(&d_y, N*sizeof(float));
    gpuMalloc(&d_out, N*sizeof(float));
    gettelapsed(talloc);
    timings.insert({std::string("allocation"), telapsed});
    auto th2d = NewTimer();
    gpuMemcpy(d_x, x, N*sizeof(float), gpuMemcpyHostToDevice);
    gpuMemcpy(d_y, y, N*sizeof(float), gpuMemcpyHostToDevice);
    gettelapsed(th2d);
    timings.insert({std::string("tH2D"), telapsed});
    ///\todo need to update kernel launch to use something other 
    /// than <<<1,1>>>
    auto tk = NewTimer();
    int blockSize, gridSize;
    // Number of threads in each thread block
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = static_cast<int>(ceil(static_cast<float>(N)/blockSize));
    // Execute the kernel
    vector_add<<<dim3(gridSize),dim3(blockSize)>>>(d_out, d_x, d_y, N);
    gettelapsed(tk);
    timings.insert({std::string("kernel"), telapsed});
    auto td2h = NewTimer();
    gpuMemcpy(out, d_out, N*sizeof(float), gpuMemcpyDeviceToHost);
    gettelapsed(td2h);
    timings.insert({std::string("tD2H"), telapsed});
    auto tfree = NewTimer();
    gpuFree(d_x);
    gpuFree(d_y);
    gpuFree(d_out);
    gettelapsed(tfree);
    timings.insert({std::string("free"), telapsed});
#endif
    delete[] x;
    delete[] y;
    delete[] out;

    return timings;
}
