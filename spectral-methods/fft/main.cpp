#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// externs
void fft_serial(float* real_op, float* imag_op, int N,
                const float* real_weight, const float* imag_weight);
void fft_parallel(float* real_op, float* imag_op, int N,
                  const float* real_weight, const float* imag_weight, int nthreads);

// timing
static inline double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// pi without relying on M_PI macro
static inline double PI() { return std::acos(-1.0); }

// -------- power-of-two / log2 helpers (match your OCL host style) --------
static inline int ilog2_int(int N) { int k=N, i=0; while (k) { k >>= 1; ++i; } return i-1; }
static inline bool is_pow2(int x) { return x > 0 && (x & (x-1)) == 0; }

// -------- bit-reversal helpers (same semantics as your ordina()/reverse) ---
static int bit_reverse_index(int N, int n) {
    int L = ilog2_int(N), p = 0;
    for (int j = 1; j <= L; ++j) {
        if (n & (1 << (L - j))) p |= 1 << (j - 1);
    }
    return p;
}

static void bit_reverse_permute(std::vector<float>& xr, std::vector<float>& xi, int N) {
    std::vector<float> tr(N), ti(N);
    for (int i = 0; i < N; ++i) {
        int r = bit_reverse_index(N, i);
        tr[i] = xr[r]; ti[i] = xi[r];
    }
    xr.swap(tr); xi.swap(ti);
}

// -------- twiddle generation (W[k] = exp(-2π i k / N), k=0..N/2-1) -------
static void generate_twiddles(int N, vector<float>& Wre, vector<float>& Wim) {
    Wre.resize(N/2);
    Wim.resize(N/2);
    const double two_pi_over_N = 2.0 * PI() / double(N);
    for (int k = 0; k < N/2; ++k) {
        double ang = -two_pi_over_N * double(k);
        Wre[k] = static_cast<float>(std::cos(ang));
        Wim[k] = static_cast<float>(std::sin(ang));
    }
}

// -------- triple-sine input -----------------------------------------------
static void gen_signal(vector<float>& xr, vector<float>& xi, int N) {
    xr.resize(N);
    xi.resize(N);
    const double s1 = 100.0 * PI() / double(N);
    const double s2 = 1000.0 * PI() / double(N);
    const double s3 = 2000.0 * PI() / double(N);
    for (int n = 0; n < N; ++n) {
        float v = static_cast<float>(std::sin(s1 * n)
                                   + std::sin(s2 * n)
                                   + std::sin(s3 * n));
        xr[n] = v;
        xi[n] = 0.0f;
    }
}

// -------- utility for correctness -----------------------------------------
static float max_abs_diff(const vector<float>& a, const vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// -------- correctness: N = 4096, 8192, 16384 ------------------------------
static void run_correctness() {
    const vector<int> Ns = {4096, 8192, 16384};
    const float tol = 5e-4f;

    for (int N : Ns) {
        if (!is_pow2(N)) { std::cerr << "N must be a power of two\n"; return; }

        vector<float> inr, ini;
        gen_signal(inr, ini, N);

        // bit-reversal to match your OpenCL host precondition
        bit_reverse_permute(inr, ini, N);

        vector<float> Wre, Wim;
        generate_twiddles(N, Wre, Wim);

        // serial baseline
        vector<float> sr = inr, si = ini;
        fft_serial(sr.data(), si.data(), N, Wre.data(), Wim.data());

        // “parallel” (same as serial for now)
        vector<int> threads = {2,4,8,16,32,64};
        cout << "N=" << N << " correctness vs serial (tol=" << tol << ")\n";
        for (int t : threads) {
            vector<float> pr = inr, pi = ini;
            fft_parallel(pr.data(), pi.data(), N, Wre.data(), Wim.data(), t);
            float er = max_abs_diff(pr, sr);
            float ei = max_abs_diff(pi, si);
            float e  = std::max(er, ei);
            cout << "  nthreads=" << t << "  max_abs_err=" << e
                 << (e <= tol ? "  OK" : "  FAIL") << "\n";
        }
        cout << "\n";
    }
}

// -------- performance: N = 4096..65536 ------------------------------------
static void run_performance() {
    const vector<int> Ns = {4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288 };
    vector<int> threads = {2,4,8,16,32,64};

    ofstream csv("fft_perf.csv");
    csv << "N,nthreads,serial_ms,parallel_ms,speedup\n";

    cout << "Performance sweep (ms):\n";
    for (int N : Ns) {
        if (!is_pow2(N)) { std::cerr << "N must be a power of two\n"; return; }

        vector<float> inr, ini;
        gen_signal(inr, ini, N);

        // bit-reversal to match your OpenCL host precondition
        bit_reverse_permute(inr, ini, N);

        vector<float> Wre, Wim;
        generate_twiddles(N, Wre, Wim);

        // serial time
        vector<float> sr = inr, si = ini;
        double t0 = now_ms();
        fft_serial(sr.data(), si.data(), N, Wre.data(), Wim.data());
        double t1 = now_ms();
        double serial_ms = t1 - t0;

        cout << "N=" << N << " | serial_ms=" << std::fixed << std::setprecision(3) << serial_ms << "\n";

        // placeholder parallel
        for (int t : threads) {
            vector<float> pr = inr, pi = ini;
            double p0 = now_ms();
            fft_parallel(pr.data(), pi.data(), N, Wre.data(), Wim.data(), t);
            double p1 = now_ms();
            double par_ms = p1 - p0;
            double speed  = serial_ms / par_ms;

            csv << N << "," << t << ","
                << std::fixed << std::setprecision(3) << serial_ms << ","
                << std::fixed << std::setprecision(3) << par_ms   << ","
                << std::fixed << std::setprecision(3) << speed    << "\n";

            cout << "  nthreads=" << std::setw(2) << t
                 << " parallel_ms=" << std::fixed << std::setprecision(3) << par_ms
                 << " speedup=" << std::fixed << std::setprecision(3) << speed << "\n";
        }
        cout << "\n";
    }
    csv.close();
    cout << "Wrote fft_perf.csv\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n"
             << "  " << argv[0] << " -c    # correctness for N=4096,8192,16384\n"
             << "  " << argv[0] << " -p    # performance for N=4096,8192,16384,32768,65536\n";
        return 1;
    }

    string mode = argv[1];
    if (mode == "-c") {
        run_correctness();
    } else if (mode == "-p") {
        run_performance();
    } else {
        cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }
    return 0;
}
