# SPMV (Sparse Matrix–Vector Multiply) — CPU

## 1) What this does
- Reads a sparse matrix in [Matrix Market format](https://networkrepository.com/mtx-matrix-market-format.html) (`.mtx`) from `../data/<matrix>/<matrix>.mtx`.
- Initialization step converts indices to **0-based** if the file is 1-based.
- Runs **y = A · x** with `x` initialized to ones (double precision). **A** is a square matrix with **n** rows, represented using compressed sparse row format. **x and y** are 1-D vectors of lenght **n** 
- Two implementations:
  - `spmv_serial`: single-thread CSR SPMV (plain C-like loops).
  - `spmv_parallel`: same logic for now; accepts `nthreads` so students can parallelize later (OpenMP target)..
- Timing measures **only** the SPMV kernel (not file I/O or CSR build).

## 2) Get the data
Use the provided script to download and extract matrices into `../data`:
```bash
chmod +x get_data.sh
./get_data.sh
```
This pulls SuiteSparse tarballs and extracts them so each matrix appears under `../data/<name>/` (for example, `../data/hollywood-2009/hollywood-2009.mtx`).
Feel free to update get_data.sh to add more matrices from https://sparse.tamu.edu/

## 3) Matrix representation (how the structs are used)
We store matrices in **CSR (Compressed Sparse Row)** form:
```cpp
struct CSR {
  int nrows;                 // number of rows
  int ncols;                 // number of columns
  std::vector<int> row_ptr;  // length nrows+1; offsets into col_idx/val
  std::vector<int> col_idx;  // column index for each nonzero
  std::vector<double> val;   // value for each nonzero
};
```
- Row `r`’s nonzeros live in `k = row_ptr[r] ... row_ptr[r+1]-1`.
- Each output entry is `y[r] = Σ_k val[k] * x[col_idx[k]]`.

## 4) How to compile
Using the provided Makefile:
```bash
make
```
This builds the `spmv` executable.


## 5) How to run with different options
**Correctness mode**
```bash
./spmv -c
```
- Loads `../data/hollywood-2009/hollywood-2009.mtx`.
- Compares `spmv_parallel` against `spmv_serial`.
- Also runs `spmv_parallel` with **1, 2, 8, 16** threads and prints time + `OK/MISMATCH`.

**Profiling mode**
```bash
./spmv -p
```
- Runs `spmv_parallel` on **every** `../data/<name>/<name>.mtx` with threads **1, 2, 4, 8, 16, 32**.
- Appends results to `../output/spmv_profile.csv`.  
  *(Note: for SPMV we do **not** compute or write MTEPS.)*

**Both**
```bash
./spmv -c -p
```

## 6) What the serial code does (`spmv_serial.cpp`)

- For each row $r$, accumulate:

  sum = Σ (from k = row_ptr[r] to row_ptr[r+1] - 1) val[k] * x[col_idx[k]]
 
  then write $y[r] = \text{sum}$.

## 7) What the parallel code should do (`spmv_parallel.cpp`)
- Currently identical to the serial version but with an `int nthreads` argument; this is the file to modify for **OpenMP** parallelization.
- The **file I/O** to parse Matrix Market into CSR.  See: [Matrix Market format](https://networkrepository.com/mtx-matrix-market-format.html).

## 8) Expected output for `-c`
Program output lines (example):
```
[-c] serial   : ../data/hollywood-2009/hollywood-2009.mtx  nrows=1139905  ncols=1139905  nnz=57515616  ms=XXXX.XXX
[-c] parallel : threads=1  ms=XXXX.XXX  OK
[-c] parallel : threads=2  ms=XXXX.XXX  OK
[-c] parallel : threads=8  ms=XXXX.XXX  OK
[-c] parallel : threads=16 ms=XXXX.XXX  OK
```
- `nrows`, `ncols` — matrix shape  
- `nnz` — number of nonzeros  
- `ms` — SPMV kernel time in milliseconds  
- `OK` means parallel matches serial within tolerance

## 9) CSV output for `-p`
Written to:
```
../output/spmv_profile.csv
```

**Header:**
```
graph,num_rows,num_cols,nnz,avg_degree,ms_t1,ms_t2,ms_t4,ms_t8,ms_t16,ms_t32,speedup_t2,speedup_t4,speedup_t8,speedup_t16,speedup_t32
```

**Column meanings:**
- `graph`: matrix name (folder under `../data`)
- `num_rows`, `num_cols`, `nnz`
- `avg_degree` = `nnz / num_rows` (average nonzeros per row)
- `ms_tK`: SPMV time with `K` threads
- `speedup_tK`: `ms_t1 / ms_tK`

**Example row (values illustrative):**
```
hollywood-2009,1139905,1139905,57515616,50.45,1234.567,890.123,612.345,456.789,380.456,350.123,1.387,2.016,2.702,3.244
```

