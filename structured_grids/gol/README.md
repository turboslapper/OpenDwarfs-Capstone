# Game of Life – Structured Grid Implementation

## Overview
This code implements **Conway’s Game of Life** on a **structured 2D grid**.  
The simulation grid is represented as an `n × m` matrix of boolean values, where each cell can be either alive (`true`) or dead (`false`).  
The structured grid implies that the spatial domain is regular, defined by rows and columns, and each cell has a fixed neighborhood relationship.  

Two implementations are provided:
- **Serial version (`gol_serial.cpp`)** – executes on a single thread.
- **Parallel version (`gol_parallel.cpp`)** – designed for multithreaded execution using `pthread`.

The driver program (`gameoflife.cpp`) orchestrates correctness tests, performance measurements, and visual mode.

---

## Project Structure
```
gameoflife.cpp        # Main entry point, handles CLI and test harness
gameoflife.h          # Shared declarations and utility functions
gol_serial.cpp/.h     # Serial implementation of Game of Life
gol_parallel.cpp/.h   # Parallel implementation (pthread-based)
config/               # Example input configurations (glider, pulsar, etc.)
```

---

## How the Code Works
1. **Input**  
   - The function `input_game` reads an ASCII configuration file.  
   - First two numbers are grid dimensions `n m`.  
   - The following characters represent the initial state (`'#'` for alive, space `' '` for dead).  
   - Allocates a `bool*` array of size `n × m`.

2. **Core Simulation**  
   - Both serial and parallel versions iterate for a specified number of steps.  
   - For each cell `(i, j)`:
     - Count the number of alive neighbors.  
     - Apply Conway’s rules:  
       - Alive if exactly 3 neighbors.  
       - Alive if already alive **and** has 2 neighbors.  
       - Otherwise dead.  

3. **Correctness Tests**  
   - Runs a suite of tests (e.g., glider, pulsar) comparing serial vs. parallel implementations.  
   - Confirms bit-for-bit identical results.

4. **Performance Tests**  
   - Runs large grids with varying thread counts.  
   - Measures wall-clock runtime and reports speedup of parallel vs. serial.

---

## Build

### OpenMP Version (CPU)
```bash
make           
make clean
```

### OpenACC Version (GPU)

**Prerequisites:**
- NVIDIA HPC SDK or PGI compiler installed
- CUDA-capable GPU

**Step-by-step build:**

1. **Load the compiler module** (if on a cluster):
   ```bash
   module load nvhpc
   # or
   module load pgi
   ```

2. **Verify compiler is available**:
   ```bash
   nvc++ --version
   # or
   pgc++ --version
   ```

3. **Compile with OpenACC flags**:
   ```bash
   nvc++ -acc -Minfo=accel -O3 -std=c++11 \
         gameoflife.cpp gol_serial.cpp gol_parallel.cpp \
         -o gameoflife
   ```

- **Correctness tests:**
  ```bash
  ./gameoflife -c
  ```

- **Performance tests:**
  ```bash
  ./gameoflife -p
  ```
---

---

## Notes
- Memory is managed with `malloc/free`, casted to C++ style for compilation.  
- The grid is a **structured 2D mesh**, so memory is contiguous and access patterns are predictable.  
- For larger scales, consider OpenMP or GPU offload for better parallel performance.  

