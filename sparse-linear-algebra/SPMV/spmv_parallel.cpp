// spmv_parallel.cpp — OpenMP parallelized SPMV implementation
#include <chrono>
#include <omp.h>

double spmv_parallel(const int* row_ptr, const int* col_idx, const double* val,
                     int nrows, const double* x, double* y, int nthreads)
{
  if(nrows<=0 || !row_ptr || !col_idx || !val || !x || !y) return 0.0;

  // Set number of threads
  omp_set_num_threads(nthreads);

  // Zero output in parallel
  #pragma omp parallel for schedule(static)
  for(int r=0; r<nrows; ++r) {
    y[r] = 0.0;
  }

  using clock = std::chrono::high_resolution_clock;
  auto t0 = clock::now();

  // Parallel CSR SpMV
  // Each row is independent, so we can parallelize over rows
  // Use static scheduling for better cache locality with large matrices
  #pragma omp parallel for schedule(static)
  for(int r=0; r<nrows; ++r){
    double sum = 0.0;
    int start = row_ptr[r];
    int end   = row_ptr[r+1];
    
    // Inner loop computes dot product for this row
    for(int k=start; k<end; ++k){
      int c = col_idx[k];
      sum += val[k] * x[c];
    }
    
    y[r] = sum;
  }

  auto t1 = clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return ms;
}