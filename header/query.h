/**
 * @file query.h
 * @author Thomas Roiseux
 * @brief Queries the devices.
 * @version 0.1
 * @date 2023-02-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef QUERY_H
#define QUERY_H

#include <fstream>

/**
 * @brief Queries all the devices.
 * @throws Cuda_exception if the device id is invalid.
 * @param file File to write the output in.
 */
void queryAll(std::ofstream *file = nullptr);

/**
 * @brief Query one device.
 * @throws Cuda_exception if the device id is invalid.
 * @param deviceId Id of the device to query.
 * @param file File to write the output in.
 */
void queryOne(int deviceId, std::ofstream *file = nullptr);

#endif // QUERY_H