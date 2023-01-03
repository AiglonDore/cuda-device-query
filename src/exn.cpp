/**
 * @file exn.cpp
 * @author Thomas Roiseux
 * @brief Implementation of exception classes.
 * @version 0.1
 * @date 2023-01-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "../header/exn.h"

Cuda_exception::Cuda_exception(cudaError_t error_id) : error_id(error_id)
{
}

const char* Cuda_exception::what() const noexcept
{
    return cudaGetErrorString((cudaError_t)error_id);
}