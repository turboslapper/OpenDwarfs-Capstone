#include <bits/stdc++.h>
using namespace std;

static inline double PI() { return acos(-1.0); }

// -------- twiddle generation (W[k] = exp(-2π i k / N), k=0..N/2-1) -------
static void generate_twiddles(int N, vector<float>& Wre, vector<float>& Wim) {
    Wre.resize(N/2);
    Wim.resize(N/2);
    const double two_pi_over_N = 2.0 * PI() / double(N);
    for (int k = 0; k < N/2; ++k) {
        double ang = -two_pi_over_N * double(k);
        Wre[k] = static_cast<float>(cos(ang));
        Wim[k] = static_cast<float>(sin(ang));
    }
}

// -------- power-of-two / log2 helpers -------------------------------------
static inline int ilog2_int(int N) { int k=N, i=0; while (k) { k >>= 1; ++i; } return i-1; }
static inline bool is_pow2(int x) { return x > 0 && (x & (x-1)) == 0; }

// -------- bit-reversal (same as your host's ordina/reverse) ----------------
static int bit_reverse_index(int N, int n) {
    int L = ilog2_int(N), p = 0;
    for (int j = 1; j <= L; ++j) {
        if (n & (1 << (L - j))) p |= 1 << (j - 1);
    }
    return p;
}

static void bit_reverse_permute(vector<float>& xr, vector<float>& xi, int N) {
    vector<float> tr(N), ti(N);
    for (int i = 0; i < N; ++i) {
        int r = bit_reverse_index(N, i);
        tr[i] = xr[r]; ti[i] = xi[r];
    }
    xr.swap(tr); xi.swap(ti);
}

// -------- triple-sine input ------------------------------------------------
static void gen_signal(vector<float>& xr, vector<float>& xi, int N) {
    xr.resize(N); xi.resize(N);
    const double s1 = 100.0  * PI() / double(N);
    const double s2 = 1000.0 * PI() / double(N);
    const double s3 = 2000.0 * PI() / double(N);
    for (int n = 0; n < N; ++n) {
        float v = static_cast<float>(sin(s1 * n) + sin(s2 * n) + sin(s3 * n));
        xr[n] = v; xi[n] = 0.0f;
    }
}

// -------- kernel-style FFT (serial; matches your OpenCL indexing) ----------
static void fft_serial(float* __restrict real_op,
                       float* __restrict imag_op,
                       int N,
                       const float* __restrict real_weight,
                       const float* __restrict imag_weight)
{
    int iters = 0; { int t = N; while (t > 1) { t >>= 1; ++iters; } }
    int n = 1;
    int a = N / 2;

    for (int j = 0; j < iters; ++j) {
        for (int i = 0; i < N; ++i) {
            if (!(i & n)) {
                // op_index = (i * a) % (n * a)   (promoted to 64-bit to avoid overflow)
                size_t op_index = ( (static_cast<size_t>(i) * static_cast<size_t>(a)) %
                                    (static_cast<size_t>(n) * static_cast<size_t>(a)) );
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

int main() {
    const int N = 524288; // 2^19
    if (!is_pow2(N)) { cerr << "N must be a power of two\n"; return 1; }

    vector<float> xr, xi;
    gen_signal(xr, xi, N);

    // Bit-reversal to match your OpenCL host’s precondition
    bit_reverse_permute(xr, xi, N);

    vector<float> Wre, Wim;
    generate_twiddles(N, Wre, Wim);

    auto t0 = chrono::high_resolution_clock::now();
    fft_serial(xr.data(), xi.data(), N, Wre.data(), Wim.data());
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1 - t0).count();
    cerr << "FFT N=" << N << " time = " << fixed << setprecision(3) << ms << " ms\n";

    // Write output CSV: real,imag per line
    ofstream csv("fft_out.csv");
    csv.setf(std::ios::fixed); csv << setprecision(9);
    for (int k = 0; k < N; ++k) {
        csv << xr[k] << "," << xi[k] << "\n";
    }
    csv.close();
    cerr << "Wrote fft_out.csv (" << N << " lines)\n";
    return 0;
}

