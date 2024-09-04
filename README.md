# CUDA Device Query

This project is a simple CUDA program designed to query and display detailed information about the CUDA-compatible devices available on your system.

## Features

- Displays key attributes of CUDA devices, such as:
  - Device name
  - Compute capability
  - Memory size
  - Clock speeds
  - Multiprocessor count
  - Other hardware specifications

## Requirements

- NVIDIA CUDA Toolkit
- Compatible GPU and drivers

## How to Build

1. Clone the repository:
   ```bash
   git clone https://github.com/AiglonDore/cuda-device-query.git
   ```
2. Compile the program using the provided Makefile:
   ```bash
   make RELEASE=TRUE
   ```

## Usage

Run the compiled binary on Linux:
```bash
./bin/device_query.out
```

Run the compiled binary on Windows:
```bash
./bin/device_query.exe
```

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE.md](./LICENSE.md) file for more details.
