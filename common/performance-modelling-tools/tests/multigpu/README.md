# Multigpu
Runs several rounds of kernel instructions concurrently on all available gpus 

## Compilation
The test uses a simple Makefile, which might need to be edited for the system it is running on. Different builds for different methods of device offloading are compiled using `buildtype=` flag when making. The different build types are 
- `omp`: OpenMP, no device offloading.
- `cuda`: CUDA build
- `hip`: HIP build 
- `cudaomp`: use Nvidia compiler to enable NVIDIA GPU OpenMP device offload
- `cudahip`: use HIP that has a CUDA backend to build 
- `hipomp`: use HIP compiler to enable NVIDIA GPU OpenMP device offload

You may need to edit the Makefile to set new builds or to alter flags/compilers appropriate to the system. There are several ways of running the code that currently require compilation with definitions. The set of instructions depends on the compilation flags. 
1. Default: allocation of memory, transfer of data from host to device, computation on device, transfer of data from device to host, free memory on device and host per interation
2. `-DRUNWITHOUTALLOC`: allocate/free memory on host/device outside of iteration, transfer of data from host to device, computation on device, transfer of data from device to host per interation
3. `-DRUNWITHOUTTRANSFER`: allocate/free memory on host/device and transfer outside of iteration computation on device, per interation


> **_NOTE:_** It is recommended to clean before building again if you have switched buildtypes. `make buildtype=${buildtype1}; make buildtype=${buildtype2} clean; make buildtype=${buildtype2};`

## Running 
Compilation will place a binary `multigpu_test.${buildtype}.exe` in the `bin/` directory. Because the code uses the library stored in the `../profile_util/lib` directory, you will need to update the `LD_LIBRARY_PATH`. Running `make` will report how you need to update the path, however, the following will work 

```bash
export LD_LIBRARY_PATH=$(pwd)/../profile_util/lib/:$LD_LIBRARY_PATH
```

The code can accept up to 2 args
1. Number of iterationsDefault is 100. 
2. How large a vector should be used in the calculation. Default is 1024*1024

The code will then report information about what parallelism is present, what devices it sees, how it is running, etc. An example output is 
```
@main L30
Parallel API's
 ========
Running with CUDA and found 2
DeviceTesla V100-PCIE-16GB Compute Units 80 Max Work Group Size 32 Local Mem Size 49152 Global Mem Size 16945512448
DeviceTesla V100-PCIE-16GB Compute Units 80 Max Work Group Size 32 Local Mem Size 49152 Global Mem Size 16945512448

@main L31
Core Binding
 ========
	 On node t019 :  Core affinity = 0

Code using: CUDA
=================================
 DEVICE 0
Reporting times for run_kernel
Times (min,1,16,median,86,99,max) [us] and corresponding indicies = (108, 165, 221, 223, 227, 586, 25229) : (12, 4610, 5532, 8771, 9733, 432, 2511)
Times (ave,stddev) [us] = (320.875, 1219.69)
---------------------------------
On device times within run_kernel
Reporting times for kernel
Times (min,1,16,median,86,99,max) [us] and corresponding indicies = (17.376, 17.376, 18.4, 18.432, 19.424, 20.48, 24702) : (3001, 8588, 5783, 9528, 8926, 7673, 4108)
Times (ave,stddev) [us] = (21.0233, 246.85)
Reporting times for tD2H
Times (min,1,16,median,86,99,max) [us] and corresponding indicies = (20.448, 21.472, 64.544, 78.848, 92.16, 133.12, 24612.9) : (2305, 7559, 6597, 9802, 209, 4572, 6423)
Times (ave,stddev) [us] = (117.378, 738.769)
Reporting times for tH2D
Times (min,1,16,median,86,99,max) [us] and corresponding indicies = (32.736, 33.76, 77.792, 90.112, 104.48, 116.736, 25074.7) : (4516, 2759, 8386, 8268, 5218, 1214, 2511)
Times (ave,stddev) [us] = (135.828, 815.064)
---------------------------------
=================================
 DEVICE 1
Reporting times for run_kernel
Times (min,1,16,median,86,99,max) [us] and corresponding indicies = (112, 113, 222, 223, 236, 675, 25198) : (9916, 9849, 2579, 3376, 8862, 98, 2473)
Times (ave,stddev) [us] = (326.813, 1221.07)
---------------------------------
On device times within run_kernel
Reporting times for kernel
Times (min,1,16,median,86,99,max) [us] and corresponding indicies = (17.376, 18.4, 18.432, 18.464, 19.456, 21.472, 4498.43) : (5576, 3671, 4881, 1334, 2481, 3449, 924)
Times (ave,stddev) [us] = (19.6744, 51.699)
Reporting times for tD2H
Times (min,1,16,median,86,99,max) [us] and corresponding indicies = (21.472, 22.528, 64.512, 78.816, 91.136, 135.136, 25034.7) : (9870, 9785, 8167, 6059, 6163, 3524, 2473)
Times (ave,stddev) [us] = (137.819, 918.496)
Reporting times for tH2D
Times (min,1,16,median,86,99,max) [us] and corresponding indicies = (33.76, 33.824, 76.768, 90.112, 102.432, 145.376, 25013.2) : (9726, 9662, 2462, 5356, 7957, 3453, 9229)
Times (ave,stddev) [us] = (130.601, 802.608)
---------------------------------
```

## Future
I will add a Python notebook that parses this output for comparison and also added OpenACC and OpenCL tests. 
