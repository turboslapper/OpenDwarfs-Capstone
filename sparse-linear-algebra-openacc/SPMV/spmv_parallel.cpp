// spmv_parallel.cpp — OpenACC-parallelized CSR SpMV implementation.
// Parallelizes over rows; 'nthreads' is mapped to OpenACC num_gangs (if > 0).

#include <chrono>
#include <algorithm>
#include <openacc.h>

double spmv_parallel(const int* row_ptr, const int* col_idx, const double* val,
                     int nrows, const double* x, double* y, int nthreads)
{
  if(nrows <= 0 || !row_ptr || !col_idx || !val || !x || !y) return 0.0;

  int nnz = row_ptr[nrows];
  if(nnz <= 0){
    for(int r=0; r<nrows; ++r) y[r] = 0.0;
    return 0.0;
  }

  // We don't get ncols from main.cpp, so compute how much of x we actually touch.
  int max_col = 0;
  for(int k=0; k<nnz; ++k) max_col = std::max(max_col, col_idx[k]);
  int x_len = max_col + 1; // nnz>0 => x_len >= 1

  using clock = std::chrono::high_resolution_clock;
  double ms = 0.0;

  // Put arrays on device once; avoid returns inside this block.
  #pragma acc data copyin(row_ptr[0:nrows+1], col_idx[0:nnz], val[0:nnz], x[0:x_len]) \
                   copyout(y[0:nrows])
  {
    // Zero output (not included in timing)
    if(nthreads > 0){
      #pragma acc parallel loop gang num_gangs(nthreads)
      for(int r=0; r<nrows; ++r) y[r] = 0.0;
    } else {
      #pragma acc parallel loop gang
      for(int r=0; r<nrows; ++r) y[r] = 0.0;
    }
    acc_wait_all();

    auto t0 = clock::now();

    // CSR SpMV
    if(nthreads > 0){
      #pragma acc parallel loop gang num_gangs(nthreads) present(row_ptr, col_idx, val, x, y)
      for(int r=0; r<nrows; ++r){
        double sum = 0.0;
        int start = row_ptr[r];
        int end   = row_ptr[r+1];

        #pragma acc loop seq
        for(int k=start; k<end; ++k){
          int c = col_idx[k];
          sum += val[k] * x[c];
        }
        y[r] = sum;
      }
    } else {
      #pragma acc parallel loop gang present(row_ptr, col_idx, val, x, y)
      for(int r=0; r<nrows; ++r){
        double sum = 0.0;
        int start = row_ptr[r];
        int end   = row_ptr[r+1];

        #pragma acc loop seq
        for(int k=start; k<end; ++k){
          int c = col_idx[k];
          sum += val[k] * x[c];
        }
        y[r] = sum;
      }
    }

    acc_wait_all();
    auto t1 = clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  return ms;
}
