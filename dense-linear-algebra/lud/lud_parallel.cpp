#include <vector>
#include <cmath>
#include <stdexcept>


// OpenACC version
void lu_decompose_parallel(std::vector<std::vector<double>>& A) {
    const int N = (int)A.size();
    const double eps = 1e-30;
    
    // convert 2D vector to 1D array for OpenACC
    double* A_flat = new double[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A_flat[i * N + j] = A[i][j];
        }
    }
    
    #pragma acc data copy(A_flat[0:N*N])
    {
        for (int k = 0; k < N; ++k) {
            double pivot;
            
            // copy pivot from GPU to host
            #pragma acc update host(A_flat[k*N+k:1])
            pivot = A_flat[k * N + k];
            
            if (std::abs(pivot) < eps) {
                throw std::runtime_error("Zero/tiny pivot in lu_decompose_parallel");
            }
            
            // parallelize division by pivot
            #pragma acc parallel loop present(A_flat)
            for (int i = k + 1; i < N; ++i) {
                A_flat[i * N + k] /= pivot;
            }
            
            // parallelize submatrix update with gang/vector
            #pragma acc parallel loop gang present(A_flat)
            for (int i = k + 1; i < N; ++i) {
                const double Lik = A_flat[i * N + k];
                #pragma acc loop vector
                for (int j = k + 1; j < N; ++j) {
                    A_flat[i * N + j] -= Lik * A_flat[k * N + j];
                }
            }
        }
    }
    
    // copy back to 2D vector
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = A_flat[i * N + j];
        }
    }
    
    delete[] A_flat;
}
/*
// OpenMP version
void lu_decompose_parallel(std::vector<std::vector<double>>& A) {
    const int N = (int)A.size();
    const double eps = 1e-30;
    
    for (int k = 0; k < N; ++k) {
        const double pivot = A[k][k];
        if (std::abs(pivot) < eps) {
            throw std::runtime_error("Zero/tiny pivot in lu_decompose_parallel (identical impl)");
        }
        // split rows
        #pragma omp parallel for schedule(static), num_threads(32)
        for (int i = k + 1; i < N; ++i) {
            A[i][k] /= pivot;
        }
        // split the updating of submatrix rows
        #pragma omp parallel for schedule(static), num_threads(32)
        for (int i = k + 1; i < N; ++i) {
            const double Lik = A[i][k];
            for (int j = k + 1; j < N; ++j) {
                A[i][j] -= Lik * A[k][j];
            }
        }
    }
}*/


