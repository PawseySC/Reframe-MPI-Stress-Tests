#include <multiGPU.h>
#include <iostream>
#include <cmath>

#if defined(USEHIP) || defined(USECUDA)
/// \defgroup kernels
//@{
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
//@}
#endif

/// \defgroup inlines
/// inline functions that are useful for abstracting some of the desired API away. 
//@{
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

inline void SetValues(int N, int offset, float *x, float *y)
{
    for (int i = 0; i < N; i++) 
    {
        x[i] = 1.0*offset;
        y[i] = 2.0*offset;
    }
}

inline std::tuple<float *, float *, float *> allocate_mem_on_device(int N) 
{
    float *d_x, *d_y, *d_out;
#if defined(USEHIP) || defined(USECUDA)
    auto nbytes = N*sizeof(float);
#ifdef NOASYNC
    gpuMalloc(&d_x, nbytes); 
    gpuMalloc(&d_y, nbytes);
    gpuMalloc(&d_out, nbytes);
#else
    gpuHostAlloc(&d_x, nbytes); 
    gpuHostAlloc(&d_y, nbytes);
    gpuHostAlloc(&d_out, nbytes);
#endif
#else 
    d_x = d_y = d_out = nullptr;
#endif
    return std::make_tuple(
        std::move(d_x), 
        std::move(d_y), 
        std::move(d_out)
    );
}

inline std::tuple<float *, float *, float *> allocate_mem_on_host(int N) 
{
    float *x, *y, *out;
    x = new float[N];
    y = new float[N];
    out = new float[N];
    return std::make_tuple(
        std::move(x), 
        std::move(y), 
        std::move(out)
    );
}

inline void deallocate_mem_on_host(float *&x, float *&y, float *&out) 
{
    delete[] x;
    delete[] y;
    delete[] out;
}

inline void deallocate_mem_on_device(float *&d_x, float *&d_y, float *&d_out) 
{
#if defined(USEHIP) || defined(USECUDA)
#ifdef NOASYNC
    gpuFree(d_x);
    gpuFree(d_y);
    gpuFree(d_out);
#else
    // gpuHostFree(d_x);
    // gpuHostFree(d_y);
    // gpuHostFree(d_out);
    gpuFreeAsync(d_x);
    gpuFreeAsync(d_y);
    gpuFreeAsync(d_out);
#endif
#endif
}

inline void transfer_host_device(int N, float *x, float *y, float *d_x, float *d_y) 
{
#if defined(USEHIP) || defined(USECUDA)
#ifdef NOASYNC
    gpuMemcpy(d_x, x, N*sizeof(float), gpuMemcpyHostToDevice);
    gpuMemcpy(d_y, y, N*sizeof(float), gpuMemcpyHostToDevice);
#else 
    gpuMemcpyAsync(d_x, x, N*sizeof(float), gpuMemcpyHostToDevice);
    gpuMemcpyAsync(d_y, y, N*sizeof(float), gpuMemcpyHostToDevice);
#endif
#endif
}
//@}

void run_on_devices(Logger &logger, int Niter, int N)
{
    std::vector<std::thread> threads;
    int deviceCount = GetNumDevices();
    struct AllTimes {
        std::vector<double> times;
        std::map<std::string, std::vector<double>> device_times;
        std::vector<double> x;
    };
    std::vector<AllTimes> alltimes(deviceCount);

    // set thread/block parallelism
    int blockSize, gridSize;
    // Number of threads in each thread block
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = static_cast<int>(ceil(static_cast<float>(N)/blockSize));

    // if using CUDA or HIP, use c++ threads to run gpu instructions asynchronously
    // for true multi-gpu usage
#if defined(USECUDA) || defined(USEHIP) || defined(_OPENACC)
    for (unsigned int device_id = 0; device_id < deviceCount; device_id++)
    {
        threads.push_back (std::thread ([&,device_id] () 
        {
            SetDevice(device_id);
#if (defined(RUNWITHOUTALLOC) || defined(RUNWITHOUTTRANSFER))
            auto [x, y, out] = allocate_mem_on_host(N);
            auto [d_x, d_y, d_out] = allocate_mem_on_device(N);
#if defined(_OPENACC)
            #pragma acc data copyin(x[:N],y[:N]) copyout(out[:N])
            {
#endif
#endif
            for (auto j=0;j<Niter;j++) 
            {
                auto t = NewTimer();
#if defined(RUNWITHOUTALLOC)
                auto timings = run_kernel_without_allocation(device_id*Niter+j, N, gridSize, blockSize, 
                x, y, out, d_x, d_y, d_out);
#elif defined(RUNWITHOUTTRANSFER)
                auto timings = run_kernel_without_transfer(device_id*Niter+j, N, gridSize, blockSize, 
                x, y, out, d_x, d_y, d_out);
#else
                auto timings = run_kernel(device_id*Niter+j, N, gridSize, blockSize);
#endif
                alltimes[device_id].times.push_back(t.get());
                for (auto &t:timings) 
                {
                    alltimes[device_id].device_times[t.first].push_back(t.second);
                }
            }
#if (defined(RUNWITHOUTALLOC) || defined(RUNWITHOUTTRANSFER))
#ifdef _OPENACC 
            }
#endif
            deallocate_mem_on_host(x, y, out);
            deallocate_mem_on_device(d_x, d_y, d_out);
#endif
        }
        ));
    }
    // join threads having launched stuff on gpus 
    for (auto &thread: threads) thread.join ();
#endif

    // run the openmp version of multi gpu 
#if defined(_OPENMP)
    #pragma omp parallel num_threads(deviceCount)
    {
#if defined(RUNWITHOUTALLOC) || defined(RUNWITHOUTTRANSFER)
        auto [x, y, out] = allocate_mem_on_host(N);
        auto [d_x, d_y, d_out] = allocate_mem_on_device(N);
#endif
        auto tid = omp_get_thread_num();
        SetDevice(tid);
#if (defined(RUNWITHOUTALLOC) || defined(RUNWITHOUTTRANSFER))
#if defined(_OPENMP)
        #pragma omp target data map(to:x[:N],y[:N]) map(from:out[:N])
#endif
// #if defined(_OPENACC)
//         #pragma acc data copyin(x[:N],y[:N]) copyout(out[:N])
// #endif
#endif
        {
            for (auto j=0;j<Niter;j++) 
            {
                auto t = NewTimer();
#if defined(RUNWITHOUTALLOC)
                auto timings = run_kernel_without_allocation(tid*Niter+j, N, gridSize, blockSize, 
                    x, y, out, d_x, d_y, d_out);
#elif defined(RUNWITHOUTTRANSFER)
                auto timings = run_kernel_without_transfer(tid*Niter+j, N, gridSize, blockSize, 
                    x, y, out, d_x, d_y, d_out);
#else
                auto timings = run_kernel(tid*Niter+j, N, gridSize, blockSize);
#endif
                alltimes[tid].times.push_back(t.get());
                for (auto &t:timings)
                {
                    alltimes[tid].device_times[t.first].push_back(t.second);
                }
            }
        }
#if defined(RUNWITHOUTALLOC) || defined(RUNWITHOUTTRANSFER)
        deallocate_mem_on_host(x, y, out);
        deallocate_mem_on_device(d_x, d_y, d_out);
#endif
    }
#endif

// #ifdef _OPENACC
//     std::cout<<"Multigpu pure OpenACC still not implemented. Exiting"<<std::endl;
//     exit(9);
// #endif

    for (unsigned int i = 0; i < alltimes.size(); i++)
    {
        std::cout<<"================================="<<std::endl;
        std::cout<<" DEVICE "<<i<<std::endl;
        logger.ReportTimes("run_kernel", alltimes[i].times);
        std::cout<<"---------------------------------"<<std::endl;
        std::cout<<"On device times within run_kernel"<<std::endl;
        for (auto &t:alltimes[i].device_times) logger.ReportTimes(t.first,t.second);
        std::cout<<"---------------------------------"<<std::endl;
    }
}

#define gettelapsed(t1)  telapsed = GetTimeTakenOnDevice(t1,__func__, std::to_string(__LINE__));

std::map<std::string, double> run_kernel(int offset, int N, int gridSize, int blockSize) 
{
    std::map<std::string, double> timings;
    float telapsed;
    auto [x, y, out] = allocate_mem_on_host(N);
    SetValues(N, offset, x, y);
#ifdef _OPENMP 
    //running with openmp 
    auto tall = NewTimer();
    #pragma omp target data map(to:x[:N],y[:N]) map(from:out[:N]) 
    {
        #pragma omp target
        #pragma omp parallel for
        for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    }
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("omp_target"), telapsed});
#elif defined(_OPENACC)
    // running with openacc
    auto tall = NewTimer();
    #pragma acc data copyin(x[:N],y[:N]) copyout(out[:N])
    {
        #pragma acc parallel loop gang vector 
        for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    }
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("acc_target"), telapsed});
#elif defined(USEHIP) || defined(USECUDA)
    // running with cuda/hip
    auto talloc = NewTimer();
    auto [d_x, d_y, d_out] = allocate_mem_on_device(N);
    gettelapsed(talloc);
    timings.insert({std::string("alloc"), telapsed});

    SetValues(N, offset, x, y);
    auto th2d = NewTimer();
    transfer_host_device(N, x,y,d_x,d_y);
    gettelapsed(th2d);
    timings.insert({std::string("tH2D"), telapsed});

    auto tk = NewTimer();
    // Execute the kernel
    vector_add<<<dim3(gridSize), dim3(blockSize)>>>(d_out, d_x, d_y, N);
    gettelapsed(tk);
    timings.insert({std::string("kernel"), telapsed});

    auto td2h = NewTimer();
#ifdef NOASYNC
    gpuMemcpy(out, d_out, N*sizeof(float), gpuMemcpyDeviceToHost);
#else
    gpuMemcpyAsync(out, d_out, N*sizeof(float), gpuMemcpyDeviceToHost);
#endif
    gettelapsed(td2h);
    timings.insert({std::string("tD2H"), telapsed});

    auto tfree = NewTimer();
    deallocate_mem_on_device(d_x, d_y, d_out);
    gettelapsed(tfree);
    timings.insert({std::string("tfree"), telapsed});
#endif

    deallocate_mem_on_host(x,y,out);
    return timings;
}

std::map<std::string, double> run_kernel_without_allocation(
    int offset, int N, int gridSize, int blockSize,
    float *&x, float *&y, float *&out, float *&d_x, float *&d_y, float *&d_out
)
{
    std::map<std::string, double> timings;
    float telapsed;
    SetValues(N, offset, x, y);
#ifdef _OPENMP 
    SetValues(N, offset, x, y);
    auto tall = NewTimer();
    #pragma omp target update to(x[:N],y[:N])
    #pragma omp target
    #pragma omp teams distribute parallel for 
    for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    #pragma omp target update from(out[:N])
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("omp_target_no_alloc"), telapsed});
#elif defined(_OPENACC)
    auto tall = NewTimer();
    #pragma acc update device(x[:N], y[:N])
    #pragma acc parallel loop gang vector 
    for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    #pragma acc update self(out[:N])
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("acc_target_no_alloc"), telapsed});

#elif defined(USEHIP) || defined(USECUDA)
    auto th2d = NewTimer();
    transfer_host_device(N, x,y,d_x,d_y);
    gettelapsed(th2d);
    timings.insert({std::string("tH2D"), telapsed});

    auto tk = NewTimer();
    // Execute the kernel
    vector_add<<<dim3(gridSize), dim3(blockSize)>>>(d_out, d_x, d_y, N);
    gettelapsed(tk);
    timings.insert({std::string("kernel"), telapsed});

    auto td2h = NewTimer();
#ifdef NOASYNC
    gpuMemcpy(out, d_out, N*sizeof(float), gpuMemcpyDeviceToHost);
#else
    gpuMemcpyAsync(out, d_out, N*sizeof(float), gpuMemcpyDeviceToHost);
#endif
    gettelapsed(td2h);
    timings.insert({std::string("tD2H"), telapsed});
#endif
    return timings;
}

std::map<std::string, double> run_kernel_without_transfer(
    int offset, int N, int gridSize, int blockSize,
    float *&x, float *&y, float *&out, float *&d_x, float *&d_y, float *&d_out
)
{
    std::map<std::string, double> timings;
    float telapsed;
#ifdef _OPENMP 
    auto tall = NewTimer();
    #pragma omp target
    #pragma omp teams distribute parallel for
    for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("omp_target_no_transfer"), telapsed});
#elif defined(_OPENACC)
    auto tall = NewTimer();
    #pragma acc parallel loop gang vector
    for (int i=0;i<N;i++) out[i] = x[i] + y[i];
    telapsed = GetTimeTaken(tall,__func__, std::to_string(__LINE__));
    timings.insert({std::string("acc_target"), telapsed});
#elif defined(USEHIP) || defined(USECUDA)
    auto tk = NewTimer();
    // Execute the kernel
    vector_add<<<dim3(gridSize), dim3(blockSize)>>>(d_out, d_x, d_y, N);
    gettelapsed(tk);
    timings.insert({std::string("kernel"), telapsed});
#endif
    return timings;
}
