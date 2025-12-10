#include <vector>
#include <cmath>
#include <stdexcept>

// identical to serial 
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
}

