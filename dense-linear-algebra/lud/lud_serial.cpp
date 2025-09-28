#include <vector>
#include <cmath>
#include <stdexcept>

// In-place LU (no pivoting) on a 2D vector matrix.
// On return, A holds packed [L|U]: U on/above diag; L below diag with unit diag implied.
void lu_decompose_serial(std::vector<std::vector<double>>& A) {
    const int N = (int)A.size();
    const double eps = 1e-30;

    for (int k = 0; k < N; ++k) {
        const double pivot = A[k][k];
        if (std::abs(pivot) < eps) {
            throw std::runtime_error("Zero/tiny pivot in lu_decompose_serial");
        }

        // Column scaling: L(i,k) = A(i,k) / U(k,k)
        for (int i = k + 1; i < N; ++i) {
            A[i][k] /= pivot;
        }

        // Trailing update: A(i,j) -= L(i,k) * U(k,j)
        for (int i = k + 1; i < N; ++i) {
            const double Lik = A[i][k];
            for (int j = k + 1; j < N; ++j) {
                A[i][j] -= Lik * A[k][j];
            }
        }
    }
}

