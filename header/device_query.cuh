/**
 * @file device_query.cuh
 * @author Thomas Roiseux
 * @brief Device query to check if the GPU is available and if the GPU is compatible with the CUDA version. Returns all information about the version.
 * @version 0.1
 * @date 2023-01-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef DEVICE_QUERY_H
#define DEVICE_QUERY_H

#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

/**
 * @brief Query the device and return the information in a string
 * 
 * @param device The device number
 * @param output The output string
 */
void deviceQuery(int device, std::string& output);

#endif // DEVICE_QUERY_H