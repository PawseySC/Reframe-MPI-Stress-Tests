# Warmup
Runs a few simple instructions for a certain set of rounds and then runs a larger set of device instructions labelled as the run kernel for a certain number of iteraions. 

## Compilation
The test uses a simple Makefile, which might need to be edited for the system it is running on. Different builds for different methods of device offloading are compiled using `buildtype=` flag when making. The different build types are 
- `omp`: OpenMP, no device offloading.
- `cuda`: CUDA build
- `hip`: HIP build 
- `cudaomp`: use Nvidia compiler to enable NVIDIA GPU OpenMP device offload
- `cudahip`: use HIP that has a CUDA backend to build 
- `hipomp`: use HIP compiler to enable NVIDIA GPU OpenMP device offload

> **_NOTE:_** It is recommended to clean before building again if you have switched buildtypes. `make buildtype=${buildtype1}; make buildtype=${buildtype2} clean; make buildtype=${buildtype2};`

## Running 
Compilation will place a binary `warm_up_test.${buildtype}.exe` in the `bin/` directory. Because the code uses the library stored in the `../profile_util/lib` directory, you will need to update the `LD_LIBRARY_PATH`. Running `make` will report how you need to update the path, however, the following will work 

```bash
export LD_LIBRARY_PATH=$(pwd)/../profile_util/lib/:$LD_LIBRARY_PATH
```

The code can accept up to three args
1. Number of rounds of warm-up. Default is 2.
2. Number of iterations of the full kernel to run. Default is 100. 
3. How to run the warm up kernel. Default is 0.
  - 0: run each simple device instruction for N rounds
  - 1: run rounds of kernel instructions going from a kernel launch, alloc, host to device, device to host. 
  - 2: run rounds of kernel instructions: alloc, host to device, device to host, kernel launch.
  - 3: run rounds of kernel instructions: host to device, device to host, kernel launch. alloc.
  - 4: run rounds of kernel instructions: device to host, kernel launch, alloc, host to device.

The code will then report information about what parallelism is present, what devices it sees, how it is running, etc. An example output is 
```
@main L30
Parallel API's
 ========
Running with HIP and found 2
HIP Device Compute Units 120 Max Work Group Size 64 Local Mem Size 65536 Global Mem Size 34342961152
HIP Device Compute Units 120 Max Work Group Size 64 Local Mem Size 65536 Global Mem Size 34342961152

@main L31
Core Binding
 ========
	 On node mi02 :  Core affinity = 0

Code using: HIP
@main L42  currently running :
2 rounds of warmup
Warming up by running each type of device instruction for the number of rounds indicated
100 iterations of  the vector add run_kernel
warmup_kernel_over_rounds running
KernelLaunchOnly on device 0 round 0 ::Time taken on device between : @launch_warmup_kernel L70 - @launch_warmup_kernel L45 : 774 [us]
KernelLaunchOnly on device 0 round 1 ::Time taken on device between : @launch_warmup_kernel L70 - @launch_warmup_kernel L45 : 17 [us]
KernelLaunchOnly on device 1 round 0 ::Time taken on device between : @launch_warmup_kernel L70 - @launch_warmup_kernel L45 : 738 [us]
KernelLaunchOnly on device 1 round 1 ::Time taken on device between : @launch_warmup_kernel L70 - @launch_warmup_kernel L45 : 22 [us]
MemAllocOnly on device 0 round 0 ::Time taken on device between : @launch_warmup_kernel L89 - @launch_warmup_kernel L77 : 222 [us]
MemAllocOnly on device 0 round 1 ::Time taken on device between : @launch_warmup_kernel L89 - @launch_warmup_kernel L77 : 120 [us]
MemAllocOnly on device 1 round 0 ::Time taken on device between : @launch_warmup_kernel L89 - @launch_warmup_kernel L77 : 136 [us]
MemAllocOnly on device 1 round 1 ::Time taken on device between : @launch_warmup_kernel L89 - @launch_warmup_kernel L77 : 124 [us]
tH2D on device 0 round 0 ::Time taken on device between : @launch_warmup_kernel L109 - @launch_warmup_kernel L107 : 1923 [us]
tH2D on device 0 round 1 ::Time taken on device between : @launch_warmup_kernel L109 - @launch_warmup_kernel L107 : 2024 [us]
tH2D on device 1 round 0 ::Time taken on device between : @launch_warmup_kernel L109 - @launch_warmup_kernel L107 : 1757 [us]
tH2D on device 1 round 1 ::Time taken on device between : @launch_warmup_kernel L109 - @launch_warmup_kernel L107 : 931 [us]
tD2H on device 0 round 0 ::Time taken on device between : @launch_warmup_kernel L132 - @launch_warmup_kernel L130 : 1482 [us]
tD2H on device 0 round 1 ::Time taken on device between : @launch_warmup_kernel L132 - @launch_warmup_kernel L130 : 744 [us]
tD2H on device 1 round 0 ::Time taken on device between : @launch_warmup_kernel L132 - @launch_warmup_kernel L130 : 2118 [us]
tD2H on device 1 round 1 ::Time taken on device between : @launch_warmup_kernel L132 - @launch_warmup_kernel L130 : 1150 [us]
Time taken between : @warmup_kernel_over_rounds L199 - @warmup_kernel_over_rounds L174 : 1.026 [s]
=================================
 DEVICE 0
Reporting times for run_kernel
Times (min,16,median,86,max) [us] and corresponding indicies = (4971, 5377, 5954, 7709, 8271) : (98, 58, 15, 42, 22)
Times (ave,stddev) [us] = (6427.8, 1060.47)
---------------------------------
On device times within run_kernel
Reporting times for allocation
Times (min,16,median,86,max) [us] and corresponding indicies = (162.24, 172.64, 182.24, 190.88, 207.998) : (98, 74, 26, 1, 0)
Times (ave,stddev) [us] = (181.555, 9.02734)
Reporting times for free
Times (min,16,median,86,max) [us] and corresponding indicies = (67.52, 72.48, 79.2, 85.44, 94.56) : (83, 71, 43, 11, 89)
Times (ave,stddev) [us] = (78.9775, 6.07469)
Reporting times for kernel
Times (min,16,median,86,max) [us] and corresponding indicies = (38.56, 40, 41.6, 66.4, 73.28) : (83, 61, 96, 37, 94)
Times (ave,stddev) [us] = (45.9615, 9.44526)
Reporting times for tD2H
Times (min,16,median,86,max) [us] and corresponding indicies = (1243.04, 1374.4, 1440, 1537.44, 1735.68) : (95, 79, 51, 48, 14)
Times (ave,stddev) [us] = (1447.89, 79.4699)
Reporting times for tH2D
Times (min,16,median,86,max) [us] and corresponding indicies = (1328.32, 1362.08, 1416.64, 1449.28, 3281.27) : (95, 84, 34, 5, 36)
Times (ave,stddev) [us] = (1493.2, 378.693)
---------------------------------
=================================
 DEVICE 1
Reporting times for run_kernel
Times (min,16,median,86,max) [us] and corresponding indicies = (3919, 4075, 4560, 6249, 7432) : (97, 65, 8, 38, 24)
Times (ave,stddev) [us] = (5022.99, 1011.34)
---------------------------------
On device times within run_kernel
Reporting times for allocation
Times (min,16,median,86,max) [us] and corresponding indicies = (144.16, 151.04, 159.84, 169.76, 358.72) : (95, 92, 51, 25, 24)
Times (ave,stddev) [us] = (161.848, 21.7103)
Reporting times for free
Times (min,16,median,86,max) [us] and corresponding indicies = (50.88, 59.84, 71.36, 75.84, 87.52) : (97, 68, 42, 41, 46)
Times (ave,stddev) [us] = (68.2032, 7.882)
Reporting times for kernel
Times (min,16,median,86,max) [us] and corresponding indicies = (37.6, 39.2, 40.64, 42.72, 60) : (98, 37, 96, 68, 52)
Times (ave,stddev) [us] = (41.4562, 3.62295)
Reporting times for tD2H
Times (min,16,median,86,max) [us] and corresponding indicies = (1031.36, 1058.88, 1121.6, 1287.68, 3113.6) : (97, 81, 42, 78, 29)
Times (ave,stddev) [us] = (1165.56, 220.733)
Reporting times for tH2D
Times (min,16,median,86,max) [us] and corresponding indicies = (1010.24, 1038.72, 1084.48, 1281.6, 1709.12) : (90, 74, 42, 51, 1)
Times (ave,stddev) [us] = (1121.01, 113.837)
---------------------------------
```

## Future
I will add a Python notebook that parses this output for comparison and also added OpenACC and OpenCL tests. 
