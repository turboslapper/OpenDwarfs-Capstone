// spmv_parallel.cpp — Optimized OpenACC GPU-accelerated SPMV implementation
// Uses NVIDIA HPC SDK (nvc++) for GPU offloading with data management optimizations
#include <chrono>
#include <cstring>

#ifdef _OPENACC
#include <openacc.h>
#endif

// Use restrict to help compiler with alias analysis
double spmv_parallel(const int* __restrict row_ptr,
                     const int* __restrict col_idx,
                     const double* __restrict val,
                     int nrows,
                     const double* __restrict x,
                     double* __restrict y,
                     int nthreads)
{
    if (nrows <= 0 || !row_ptr || !col_idx || !val || !x || !y || nthreads <= 0)
        return 0.0;

    // nthreads parameter ignored for GPU - OpenACC handles parallelization
    (void)nthreads;

    // Get total number of non-zeros
    const int nnz = row_ptr[nrows];
    
    // Estimate ncols from col_idx (find max + 1)
    int ncols = 0;
    for (int i = 0; i < nnz; ++i) {
        if (col_idx[i] >= ncols) ncols = col_idx[i] + 1;
    }

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

#ifdef _OPENACC
    // GPU-accelerated path using OpenACC
    // Use structured data regions for optimal memory management
    
    // Copy data to GPU (copyin for read-only, copyout for results)
    #pragma acc data copyin(row_ptr[0:nrows+1], col_idx[0:nnz], val[0:nnz], x[0:ncols]) copyout(y[0:nrows])
    {
        // Parallel loop over rows - each row is independent
        // gang = thread blocks, vector = threads within block
        // vector_length tuned for warp efficiency on NVIDIA GPUs
        #pragma acc parallel loop gang vector vector_length(128) present(row_ptr, col_idx, val, x, y)
        for (int r = 0; r < nrows; ++r) {
            const int row_start = row_ptr[r];
            const int row_end = row_ptr[r + 1];
            
            double sum = 0.0;
            
            // Inner loop reduction - compiler will optimize this
            // seq forces sequential execution within each row (sum depends on previous iteration)
            #pragma acc loop seq reduction(+:sum)
            for (int k = row_start; k < row_end; ++k) {
                sum += val[k] * x[col_idx[k]];
            }
            
            y[r] = sum;
        }
    }
    
#else
    // Fallback CPU path (serial) when OpenACC not available
    for (int r = 0; r < nrows; ++r) {
        const int row_start = row_ptr[r];
        const int row_end = row_ptr[r + 1];
        
        double sum = 0.0;
        for (int k = row_start; k < row_end; ++k) {
            sum += val[k] * x[col_idx[k]];
        }
        y[r] = sum;
    }
#endif

    auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Alternative version with explicit async data management for multiple SPMV calls
// Useful when doing iterative solvers where matrix stays on GPU
double spmv_parallel_async(const int* __restrict row_ptr,
                           const int* __restrict col_idx,
                           const double* __restrict val,
                           int nrows,
                           int ncols,
                           int nnz,
                           const double* __restrict x,
                           double* __restrict y,
                           bool first_call,
                           bool last_call)
{
    if (nrows <= 0 || !row_ptr || !col_idx || !val || !x || !y)
        return 0.0;

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

#ifdef _OPENACC
    if (first_call) {
        // First call: copy matrix structure to GPU (stays resident)
        #pragma acc enter data copyin(row_ptr[0:nrows+1], col_idx[0:nnz], val[0:nnz])
    }
    
    // Each call: copy x in, compute, copy y out
    #pragma acc data copyin(x[0:ncols]) copyout(y[0:nrows])
    {
        #pragma acc parallel loop gang vector vector_length(128) present(row_ptr, col_idx, val, x, y)
        for (int r = 0; r < nrows; ++r) {
            const int row_start = row_ptr[r];
            const int row_end = row_ptr[r + 1];
            
            double sum = 0.0;
            #pragma acc loop seq reduction(+:sum)
            for (int k = row_start; k < row_end; ++k) {
                sum += val[k] * x[col_idx[k]];
            }
            y[r] = sum;
        }
    }
    
    if (last_call) {
        // Last call: free GPU memory
        #pragma acc exit data delete(row_ptr[0:nrows+1], col_idx[0:nnz], val[0:nnz])
    }
#else
    // Fallback CPU path
    (void)first_call; (void)last_call; (void)ncols; (void)nnz;
    for (int r = 0; r < nrows; ++r) {
        double sum = 0.0;
        for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            sum += val[k] * x[col_idx[k]];
        }
        y[r] = sum;
    }
#endif

    auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
