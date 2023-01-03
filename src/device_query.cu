/**
 * @file device_query.cu
 * @author Thomas Roiseux
 * @brief Implements {@link device_query.cuh}.
 * @version 0.1
 * @date 2023-01-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "../header/device_query.cuh"

#include <sstream>

void deviceQuery(int device, std::string& output)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    std::ostringstream oss;

    oss << "Device " << device << ": " << deviceProp.name << std::endl;
    oss << "  Major revision number:                         " << deviceProp.major << std::endl;
    oss << "  Minor revision number:                         " << deviceProp.minor << std::endl;
    oss << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    oss << "  Number of multiprocessors:                     " << deviceProp.multiProcessorCount << std::endl;
    oss << "  Total amount of constant memory:               " << deviceProp.totalConstMem << " bytes" << std::endl;
    oss << "  Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    oss << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
    oss << "  Warp size:                                     " << deviceProp.warpSize << std::endl;
    oss << "  Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock << std::endl;
    oss << "  Maximum sizes of each dimension of a block:    " << deviceProp.maxThreadsDim[0] << " x "
                                                             << deviceProp.maxThreadsDim[1] << " x "
                                                             << deviceProp.maxThreadsDim[2] << std::endl;
    oss << "  Maximum sizes of each dimension of a grid:     " << deviceProp.maxGridSize[0] << " x "
                                                                << deviceProp.maxGridSize[1] << " x "
                                                                << deviceProp.maxGridSize[2] << std::endl;
    oss << "  Maximum memory pitch:                          " << deviceProp.memPitch << " bytes" << std::endl;
    oss << "  Texture alignment:                             " << deviceProp.textureAlignment << " bytes" << std::endl;
    oss << "  Clock rate:                                    " << deviceProp.clockRate << " kilohertz";
    output = oss.str();
}