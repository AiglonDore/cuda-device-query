/**
 * @file main.cpp
 * @author Thomas Roiseux
 * @brief Main file.
 * @version 0.1
 * @date 2023-01-03
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <fstream>
#include <thread>

#include "../header/device_query.cuh"
#include "../header/exn.h"

int main(int argc, char *argv[])
{
    std::cout << "\t\t-----CUDA device query-----" << std::endl;
    std::ofstream *file = nullptr;
    for (size_t i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-f") == 0)
        {
            if (i + 1 < argc)
            {
                file = new std::ofstream(argv[i + 1]);
                if (!file->is_open())
                {
                    std::cerr << "Error while opening file: " << argv[i + 1] << "." << std::endl;
                    std::cerr << "Ignoring it." << std::endl;
                    delete file;
                    file = nullptr;
                }
                else
                {
                    std::cout << "Output will be written in file: " << argv[i + 1] << "." << std::endl;
                }
                i++;
            }
            else
            {
                std::cerr << "No file specified after argument: " << argv[i] << std::endl;
                std::cerr << "Ignoring it." << std::endl;
            }
        }
        else
        {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Ignoring it." << std::endl;
        }
    }
    
    try
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
            return 0;
        }

        std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
        if (file != nullptr)
        {
            *file << "Number of CUDA devices: " << deviceCount << std::endl;
        }

        if (deviceCount == 1)
        {
            std::string result;
            deviceQuery(0, result);
            if (file != nullptr)
            {
                *file << result << std::endl;
            }
            else
            {
                std::cout << result << std::endl;
            }
        }
        else
        {
            std::cout << "Querying devices using up to " << std::thread::hardware_concurrency() << " threads." << std::endl;
            int nbThreads = std::min(deviceCount, (int)std::thread::hardware_concurrency());
            std::thread *threads = new std::thread[nbThreads];
            std::string *results = new std::string[deviceCount];

            for (int i = 0; i < nbThreads; i++)
            {
                threads[i] = std::thread([&nbThreads, &deviceCount, results](int index) {
                    for (size_t i = index; i < deviceCount; i+=nbThreads)
                    {
                        deviceQuery(i, results[i]);
                    }
                }, i);
            }

            for (int i = 0; i < nbThreads; i++)
            {
                threads[i].join();
            }
            if (file != nullptr)
            {
                for (int i = 0; i < deviceCount; i++)
                {
                    *file << results[i] << std::endl;
                }
            }
            else
            {
                for (int i = 0; i < deviceCount; i++)
                {
                    std::cout << results[i] << std::endl;
                }
            }
            delete[] threads;
            delete[] results;
        }
    }
    catch (const Cuda_exception &e)
    {
        std::cerr << "Error with CUDA" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    if (file != nullptr)
    {
        file->close();
        delete file;
        file = nullptr;
    }

    return 0;
}