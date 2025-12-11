// spmv_parallel.cpp — Optimized OpenMP SPMV implementation
#include <chrono>
#include <omp.h>
#include <vector>

// Use restrict to help the compiler with alias analysis
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

    omp_set_num_threads(nthreads);

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    // Small matrices or single thread: just run serial, but still use the same code path.
    if (nthreads == 1 || nrows < 256) {
        for (int r = 0; r < nrows; ++r) {
            const int start = row_ptr[r];
            const int end   = row_ptr[r + 1];
            const int len   = end - start;

            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

            int k = start;
            const int unroll_end = start + (len & ~7); // 8x unrolling

            // 8x unrolled inner loop with 4 accumulators
            for (; k < unroll_end; k += 8) {
                sum0 += val[k]     * x[col_idx[k]];
                sum1 += val[k + 1] * x[col_idx[k + 1]];
                sum2 += val[k + 2] * x[col_idx[k + 2]];
                sum3 += val[k + 3] * x[col_idx[k + 3]];

                sum0 += val[k + 4] * x[col_idx[k + 4]];
                sum1 += val[k + 5] * x[col_idx[k + 5]];
                sum2 += val[k + 6] * x[col_idx[k + 6]];
                sum3 += val[k + 7] * x[col_idx[k + 7]];
            }

            for (; k < end; ++k) {
                sum0 += val[k] * x[col_idx[k]];
            }

            y[r] = sum0 + sum1 + sum2 + sum3;
        }
    } else {
        // Parallel over rows. Let OpenMP handle partitioning; use dynamic scheduling
        // to cope with highly irregular row lengths (e.g., power-law graphs).
        // Tune chunk size if needed.
        const int chunk = 64;

        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, chunk)
            for (int r = 0; r < nrows; ++r) {
                const int start = row_ptr[r];
                const int end   = row_ptr[r + 1];
                const int len   = end - start;

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

                int k = start;
                const int unroll_end = start + (len & ~7);

                // Hint to the compiler: try to vectorize despite gathers
                #pragma omp simd reduction(+:sum0,sum1,sum2,sum3)
                for (int kk = k; kk < unroll_end; kk += 8) {
                    sum0 += val[kk]     * x[col_idx[kk]];
                    sum1 += val[kk + 1] * x[col_idx[kk + 1]];
                    sum2 += val[kk + 2] * x[col_idx[kk + 2]];
                    sum3 += val[kk + 3] * x[col_idx[kk + 3]];

                    sum0 += val[kk + 4] * x[col_idx[kk + 4]];
                    sum1 += val[kk + 5] * x[col_idx[kk + 5]];
                    sum2 += val[kk + 6] * x[col_idx[kk + 6]];
                    sum3 += val[kk + 7] * x[col_idx[kk + 7]];
                }

                k = unroll_end;
                for (; k < end; ++k) {
                    sum0 += val[k] * x[col_idx[k]];
                }

                y[r] = sum0 + sum1 + sum2 + sum3;
            }
        }
    }

    auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
