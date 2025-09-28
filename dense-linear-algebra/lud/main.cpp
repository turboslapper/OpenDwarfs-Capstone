#include <bits/stdc++.h>

// Prototypes (implemented in lud_serial.cpp / lud_parallel.cpp)
void lu_decompose_serial(std::vector<std::vector<double>>& A);
void lu_decompose_parallel(std::vector<std::vector<double>>& A); // to be parallelized

// Build dense 3D Laplacian (7-point, Dirichlet stencil) into a 2D matrix.
// Diagonal = number of in-grid neighbors; neighbor entries = -1.
static std::vector<std::vector<double>> make_laplacian3d_dense(int nx, int ny, int nz) {
    const int N = nx * ny * nz;
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 0.0));

    auto idx3 = [nx, ny](int x, int y, int z) -> int {
        return (z * ny + y) * nx + x;
    };

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int i = idx3(x, y, z);
                int deg = 0;
                auto link = [&](int xx, int yy, int zz) {
                    const int j = idx3(xx, yy, zz);
                    A[i][j] = -1.0;
                    deg++;
                };
                if (x > 0)       link(x - 1, y, z);
                if (x + 1 < nx)  link(x + 1, y, z);
                if (y > 0)       link(x, y - 1, z);
                if (y + 1 < ny)  link(x, y + 1, z);
                if (z > 0)       link(x, y, z - 1);
                if (z + 1 < nz)  link(x, y, z + 1);
                A[i][i] = (double)deg;
            }
        }
    }
    return A;
}

// ===== Utilities =====
static double frob(const std::vector<std::vector<double>>& A) {
    long double s = 0.0L;
    for (const auto& row : A) for (double v : row) s += (long double)v * (long double)v;
    return (double)std::sqrt(s);
}

static void unpack_LU(const std::vector<std::vector<double>>& AU,
                      std::vector<std::vector<double>>& L,
                      std::vector<std::vector<double>>& U)
{
    const int N = (int)AU.size();
    L.assign(N, std::vector<double>(N, 0.0));
    U.assign(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        L[i][i] = 1.0;
        for (int j = 0; j < N; ++j) {
            if (i > j) L[i][j] = AU[i][j];
            else       U[i][j] = AU[i][j];
        }
    }
}

static double rel_err_reconstruct(const std::vector<std::vector<double>>& AU,
                                  const std::vector<std::vector<double>>& Aorig)
{
    const int N = (int)AU.size();
    std::vector<std::vector<double>> L, U;
    unpack_LU(AU, L, U);

    std::vector<std::vector<double>> LU(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k <= i; ++k) {               // L[i,k] nonzero only for k<=i
            const double Lik = L[i][k];
            const auto& Uk = U[k];                   // U[k,j] nonzero only for j>=k
            auto& LUi = LU[i];
            for (int j = k; j < N; ++j) LUi[j] += Lik * Uk[j];
        }
    }

    std::vector<std::vector<double>> R = Aorig;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            R[i][j] -= LU[i][j];

    return frob(R) / std::max(1e-30, frob(Aorig));
}

static void diff_stats(const std::vector<std::vector<double>>& A,
                       const std::vector<std::vector<double>>& B,
                       double& max_abs, double& rms)
{
    const int N = (int)A.size();
    long double s2 = 0.0L;
    max_abs = 0.0;
    long double cnt = 0.0L;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double d = std::abs(A[i][j] - B[i][j]);
            max_abs = std::max(max_abs, d);
            s2 += (long double)d * (long double)d;
            cnt += 1.0L;
        }
    }
    rms = (double)std::sqrt((double)(s2 / std::max(1.0L, cnt)));
}

static double time_once(void(*fn)(std::vector<std::vector<double>>&),
                        std::vector<std::vector<double>> A) // by-value copy for fair timing
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn(A);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void run_compare_case(int nx, int ny, int nz) {
    const int N = nx * ny * nz;
    std::cout << "\n=== Compare nx=" << nx << " ny=" << ny << " nz=" << nz
              << "  (N=" << N << ", N×N=" << N << "×" << N << ") ===\n";
    auto A = make_laplacian3d_dense(nx, ny, nz);

    auto AU_ser = A;
    auto AU_par = A;

    lu_decompose_serial(AU_ser);
    lu_decompose_parallel(AU_par);

    double max_abs, rms;
    diff_stats(AU_ser, AU_par, max_abs, rms);
    std::cout << "Packed-LU diff (serial vs parallel-identical): "
              << "max_abs=" << std::scientific << max_abs
              << "  rms=" << rms << "\n";

    double rel_ser = rel_err_reconstruct(AU_ser, A);
    double rel_par = rel_err_reconstruct(AU_par, A);
    std::cout << "Reconstruction rel error ||A-LU||_F/||A||_F : "
              << "serial=" << rel_ser << "  parallel=" << rel_par << "\n";
}

struct PerfRow {
    int nx, ny, nz, N;
    double t_ser_ms, t_par_ms, speedup;
};

static void run_perf_and_log_csv(const std::vector<std::tuple<int,int,int>>& cases,
                                 const std::string& csv_path)
{
    std::vector<PerfRow> rows;
    rows.reserve(cases.size());

    for (auto [nx, ny, nz] : cases) {
        const int N = nx * ny * nz;
        std::cout << "\n=== Perf nx=" << nx << " ny=" << ny << " nz=" << nz
                  << "  (N=" << N << ", N×N=" << N << "×" << N << ") ===\n";

        auto A = make_laplacian3d_dense(nx, ny, nz);

        // Timings (each uses its own copy)
        const double t_ser = time_once(lu_decompose_serial,   A);
        const double t_par = time_once(lu_decompose_parallel, A); // identical impl
        const double sp = (t_par > 0.0) ? (t_ser / t_par) : std::numeric_limits<double>::infinity();

        std::cout << "  serial    = "   << std::fixed << std::setprecision(2) << t_ser << " ms\n";
        std::cout << "  parallel* = "   << std::fixed << std::setprecision(2) << t_par << " ms\n";
        std::cout << "  speedup (serial/parallel*) = " << std::setprecision(3) << sp << "x\n";

        rows.push_back({nx, ny, nz, N, t_ser, t_par, sp});
    }

    // Write CSV
    std::ofstream ofs(csv_path);
    if (!ofs) {
        std::cerr << "ERROR: could not open CSV for write: " << csv_path << "\n";
        return;
    }
    ofs << "nx,ny,nz,N,serial_ms,parallel_ms,speedup\n";
    ofs.setf(std::ios::fixed); ofs << std::setprecision(6);
    for (const auto& r : rows) {
        ofs << r.nx << ',' << r.ny << ',' << r.nz << ',' << r.N << ','
            << r.t_ser_ms << ',' << r.t_par_ms << ',' << r.speedup << '\n';
    }
    ofs.close();
    std::cout << "\nCSV written: " << csv_path << "\n";
}

static void usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " -c        # compare LU for nx=ny=nz in {8,16}\n"
              << "  " << prog << " -p|-P     # time serial vs parallel* and write lud_perf.csv for:\n"
              << "                           1) 8x8x8   (N=512)\n"
              << "                           2) 8x16x8  (N=1024)\n"
              << "                           3) 8x16x16 (N=2048)\n"
              << "                           4) 16x16x16(N=4096)\n"
              << "\nNote: parallel* is intentionally identical to serial (no OpenMP), per request.\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 0; }
    std::string mode = argv[1];
    if (mode == "-c") {
        run_compare_case(8, 8, 8);
        run_compare_case(16, 16, 16);
    } else if (mode == "-p" || mode == "-P") {
        const std::vector<std::tuple<int,int,int>> cases = {
            {8, 8, 8},      // N=512
            {8, 16, 8},     // N=1024
            {8, 16, 16},    // N=2048
            {16, 16, 16}    // N=4096
        };
        run_perf_and_log_csv(cases, "lud_perf.csv");
    } else {
        usage(argv[0]);
    }
    return 0;
}

