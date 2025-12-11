#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <cstdio>
//
// Intentionally identical to serial; students will parallelize later.
//openACC solution:
/**void fft_parallel(float* __restrict real_op,
                  float* __restrict imag_op,
                  int N,
                  const float* __restrict real_weight,
                  const float* __restrict imag_weight,
                  int nthreads)
{
    int iters = 0; { int t = N; while (t > 1) { t >>= 1; ++iters; } } //this computes number of FFT stages
    int n = 1;
    int a = N / 2;
    //copyin is data that is inputted into GPU and doesn't need to be outputted
    //copy is for data that both needs to be inputted or outputted
    #pragma acc data copyin(real_weight[0:N/2], imag_weight[0:N/2]) copy(real_op[0:N], imag_op[0:N])
    for (int j = 0; j < iters; ++j)  //can't parallelize here becuase each iteration depends on previous one
    {
        //#pragma acc data copyin(real_weight[0:N/2], imag_weight[0:N/2]) copy(real_op[0:N], imag_op[0:N])
        #pragma acc parallel loop present(real_weight[0:N/2], imag_weight[0:N/2], real_op[0:N], imag_op[0:N])
        for (int i = 0; i < N; ++i)
        {
            //n is always a power of 2
            //if statement checks the log(n)th bit of i
            if (!(i & n))
            {
                //this block actually computes the fourier transform, "returns" output in real_op and imag_op
                std::size_t op_index = ( (static_cast<std::size_t>(i) * static_cast<std::size_t>(a)) %
                                         (static_cast<std::size_t>(n) * static_cast<std::size_t>(a)) );
                int res_index = i + n;

                float real_temp = real_op[i];
                float imag_temp = imag_op[i];

                float rw = real_weight[op_index];
                float iw = imag_weight[op_index];

                float real_Temp = rw * real_op[res_index] - iw * imag_op[res_index];
                float imag_Temp = iw * real_op[res_index] + rw * imag_op[res_index];

                real_op[res_index] = real_temp - real_Temp;
                imag_op[res_index] = imag_temp - imag_Temp;
                real_op[i]         = real_temp + real_Temp;
                imag_op[i]         = imag_temp + imag_Temp;
            }
        }
        n <<= 1;
        a >>= 1;
    }
}**/
//openMP solution (not ACC)
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <cstdio>

// Intentionally identical to serial; students will parallelize later.
void fft_parallel(float* __restrict real_op,
                  float* __restrict imag_op,
                  int N,
                  const float* __restrict real_weight,
                  const float* __restrict imag_weight,
                  int nthreads)
{
    int iters = 0; { int t = N; while (t > 1) { t >>= 1; ++iters; } } //this computes number of FFT stages
    int n = 1;
    int a = N / 2;
    for (int j = 0; j < iters; ++j)  //can't parallelize here becuase each iteration depends on previous one
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int i = 0; i < N; ++i)
        {
            //n is always a power of 2
            //if statement checks the log(n)th bit of i
            if (!(i & n))
            {
                //this block actually computes the fourier transform, "returns" output in real_op and imag_op
                std::size_t op_index = ( (static_cast<std::size_t>(i) * static_cast<std::size_t>(a)) %
                                         (static_cast<std::size_t>(n) * static_cast<std::size_t>(a)) );
                int res_index = i + n;

                float real_temp = real_op[i];
                float imag_temp = imag_op[i];

                float rw = real_weight[op_index];
                float iw = imag_weight[op_index];

                float real_Temp = rw * real_op[res_index] - iw * imag_op[res_index];
                float imag_Temp = iw * real_op[res_index] + rw * imag_op[res_index];

                real_op[res_index] = real_temp - real_Temp;
                imag_op[res_index] = imag_temp - imag_Temp;
                real_op[i]         = real_temp + real_Temp;
                imag_op[i]         = imag_temp + imag_Temp;
            }
        }
        n <<= 1;
        a >>= 1;
    }
}
