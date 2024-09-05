/*! \file kernels.h
 *  \brief kernels of for warming up and running on GPU
 */

#ifndef KERNELS_H
#define KERNELS_H

#include <profile_util.h>

/// GPU kernels types definitions
//@{
#define KERNEL_DEFAULT 0
//@}

/// \defgroup kernels
/// GPU kernels
//@{
template <typename T> __global__ void vector_square(const T *A_d, T *C_d, size_t N);
void silly_template_instansiation();
void compute_kernel1(size_t N, 
    std::vector<int*> &x_int_gpu, 
    std::vector<int*> &y_int_gpu, 
    std::vector<float*> &x_float_gpu, 
    std::vector<float*> &y_float_gpu, 
    std::vector<double*> &x_double_gpu, 
    std::vector<double*> &y_double_gpu,
    int Niter = 1,
    size_t blocksize = 256,
    size_t threadsperblock = 1024
    );
//@}
#endif
