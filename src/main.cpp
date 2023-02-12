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
#include <fstream>
#include <cstring>

#include "../header/query.h"
#include "../header/exn.h"

/**
 * @brief Print the usage of the program.
 * 
 * @param name Name of the program.
 */
void print_usage(const char *name)
{
    std::cout << "Usage: " << name << " [options] [device_id]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "\t--file, -f\t\tSpecify a file to write the output in." << std::endl;
    std::cout << "\t--help, -h\t\tPrint this help." << std::endl;
    std::cout << "If no device_id is specified, all devices will be queried." << std::endl;
}

/**
 * @brief Main function.
 * 
 * @param argc Number of arguments.
 * @param argv List of arguments.
 * @return int Error code.
 */
int main(int argc, char *argv[])
{
    std::cout << "\t\t-----CUDA device query-----" << std::endl;
    std::ofstream *file = nullptr;
    int *deviceId = nullptr;

    try
    {
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
                i++;
            }
            else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            {
                print_usage(argv[0]);
                if (file != nullptr)
                {
                    file->close();
                    delete file;
                    file = nullptr;
                }
                delete deviceId;
                return 0;
            }
            else if (argv[i][0] == '-')
            {
                std::cerr << "Unknown option: " << argv[i] << std::endl;
                std::cerr << "Ignoring it." << std::endl;
            }
            else if (deviceId == nullptr)
            {
                deviceId = new int(atoi(argv[i]));
            }
            else
            {
                std::cerr << "Unknown argument: " << argv[i] << std::endl;
                std::cerr << "Ignoring it." << std::endl;
            }
        }
        if (deviceId == nullptr)
        {
            queryAll(file);
        }
        else
        {
            queryOne(*deviceId, file);
        }
        
    }
    catch (const Cuda_exception &e)
    {
        std::cerr << argv[0] << ": " << "Error with CUDA: ";
        std::cerr << e.what() << std::endl;
        if (file != nullptr)
        {
            file->close();
            delete file;
            file = nullptr;
        }
        return e.get_error_id();
    }
    catch (const std::exception &e)
    {
        std::cerr << argv[0] << ": " << e.what() << std::endl;
        if (file != nullptr)
        {
            file->close();
            delete file;
            file = nullptr;
        }
        return -1;
    }
    delete deviceId;
    if (file != nullptr)
    {
        file->close();
        delete file;
        file = nullptr;
    }
    return 0;
}