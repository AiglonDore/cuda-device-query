/**
 * @file query.cpp
 * @author Thomas Roiseux
 * @brief Implements {@link query.h}.
 * @version 0.1
 * @date 2023-02-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "../header/query.h"
#include "../header/device_query.cuh"
#include "../header/exn.h"

#include <iostream>
#include <string>


void queryAll(std::ofstream *file)
{
    int deviceCount = 0;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
        if (error_id != cudaSuccess)
        {
            throw Cuda_exception(error_id);
        }
        if (deviceCount == 0)
        {
            std::cerr << "There is no device supporting CUDA." << std::endl;
            if (file != nullptr)
            {
                file->close();
                delete file;
                file = nullptr;
            }
            exit(0);
        }

        if (file != nullptr)
        {
            *file << "Number of CUDA devices: " << deviceCount << std::endl;
        }
        else
        {
            std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
        }

        std::string result;
        for (size_t i = 0; i < deviceCount; i++)
        {
            deviceQuery(i, result);
            if (file != nullptr)
            {
                *file << result << std::endl;
            }
            else
            {
                std::cout << result << std::endl;
            }
        }
}

void queryOne(int deviceId, std::ofstream *file)
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess)
    {
        throw Cuda_exception(error_id);
    }
    if (deviceCount == 0)
    {
        std::cerr << "There is no device supporting CUDA." << std::endl;
        if (file != nullptr)
        {
            file->close();
            delete file;
            file = nullptr;
        }
        exit(0);
    }

    if (deviceId >= deviceCount)
    {
        throw Cuda_exception(cudaErrorInvalidDevice);
    }

    std::string result;
    deviceQuery(deviceId, result);
    if (file != nullptr)
    {
        *file << result << std::endl;
    }
    else
    {
        std::cout << result << std::endl;
    }
}