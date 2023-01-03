/**
 * @file exn.h
 * @author Thomas Roiseux
 * @brief Exception classes.
 * @version 0.1
 * @date 2023-01-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef EXN_H
#define EXN_H

#include <exception>
#include <stdexcept>

#include "cuda.h"
#include "cuda_runtime.h"

/**
 * @brief Exception class for CUDA errors.
 * 
 */
class Cuda_exception : public std::exception
{
private:
    cudaError_t error_id;
public:
    /**
     * @brief Construct a new cuda exception object
     * 
     * @param error_id CUDA error ID
     */
    Cuda_exception(cudaError_t error_id);
    /**
     * @brief Get the error message
     * 
     * @return const char* Error message
     */
    virtual const char* what() const noexcept override;
};

#endif // EXN_H