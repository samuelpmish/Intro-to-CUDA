# Intro to CUDA

Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

```bash
git clone https://github.com/samuelpmish/Intro-to-CUDA.git
cd Intro-to-CUDA
```

## Build with CMake

Requires CMake 3.21 or newer.

```bash
cmake -S . -B build
cmake --build build -j
```

Executables are written to `build/`.

## Build with the Makefile

```bash
make -j
```

Executables are written to `build/make/`. Build one example with `make hello_world`, or run `make list` to see all targets.
