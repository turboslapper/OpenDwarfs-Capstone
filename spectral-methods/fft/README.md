# Fast Fourier Transform (FFT)

This code implements a **radix-2 FFT** with bit-reversal

- **What it does:**  
  Performs an in-place FFT on a synthetic test signal (sum of three sine waves), stage by stage using butterfly operations and precomputed twiddle factors.  
  Provides both a serial implementation and a placeholder "parallel" version (currently identical) for students to parallelize.

---

## How to run
To run in OpenACC:
1) go into the make file and ensure
CXX      := nvc++
CXXFLAGS := -acc -gpu=cuda12.3 -O3
are uncommented and that the other CXX and CXXFLAGS are commented out
2) go into fft_parallel.cpp, there you should see //openACC solution on line 6. Ensure that the lines after that are uncommented out, and that the OpenMP code (lines 54-end) are commented out
3) run make clean then make
4) ./fft_app -c or ./fft_app -p

To run in OpenMP:
1) go into the make file and ensure
CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp -Wall -Wextra
are uncommented and that the other CXX and CXXFLAGS are commented out
2) go into fft_parallel.cpp, there you should see //openMP solution on line 54. Ensure that the lines after that are uncommented out, and that the OpenACC code (lines 7-52) are commented out
3) run make clean then make
4) ./fft_app -c or ./fft_app -p
## Input Signal

For size *N*, the input is:  

$$x[n] = \sin\left(\tfrac{100\pi}{N}n\right) + \sin\left(\tfrac{1000\pi}{N}n\right) + \sin\left(\tfrac{2000\pi}{N}n\right), \quad n = 0,1,\dots,N-1$$  



- Real part = `x[n]`  
- Imaginary part = `0`  

Before the FFT stages, the input undergoes **bit-reversal reordering**.

---

## Files

- `fft_serial.cpp` — Serial FFT implementation (butterflies as in kernel).  
- `fft_parallel.cpp` — Parallel FFT (currently same as serial; to be parallelized).  
- `main.cpp` — Driver: generates input + twiddles, applies bit-reversal, runs correctness & performance.  
- `Makefile` — Used to compile the code.

---

## Build

```bash
make
```

Produces the executable `fft_app`.

---

## Run

### Correctness checks
```bash
./fft_app -c
```
- Runs FFT for `N=4096,8192,16384`.  
- Compares serial and parallel results (max abs error).  

### Performance evaluation
```bash
./fft_app -p
```
- Runs FFT for `N=4096,8192,16384,32768,65536 ....`.  
- Tests placeholder parallel across thread counts `2..64`.  
- Outputs **fft_perf.csv** with runtime and speedup (parallel = same as serial for now).

---

## Clean

```bash
make clean
```

Removes objects, binary, and `fft_perf.csv`.

