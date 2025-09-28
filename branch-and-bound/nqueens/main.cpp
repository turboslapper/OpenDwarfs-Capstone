#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <omp.h>

// Forward declarations; implementations live in nq_serial.cpp / nq_parallel.cpp
std::uint64_t count_nqueens_serial(int n);
std::uint64_t count_nqueens_parallel(int n);

static inline double ms_between(std::chrono::steady_clock::time_point a,
                                std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void usage(const char* p) {
    std::cerr << "Usage:\n"
              << "  " << p << " -c   # correctness check at n=10\n"
              << "  " << p << " -p   # write timings CSV to ../output/results.csv for n=8,10,12,14,15 and threads 1,2,4,8,16,32\n";
}

int main(int argc, char** argv) {
    if (argc != 2) { usage(argv[0]); return 1; }
    std::string mode = argv[1];

    if (mode == "-c") {
        const int n = 10;
        const std::uint64_t expected = 724;

        auto t0 = std::chrono::steady_clock::now();
        auto s_cnt = count_nqueens_serial(n);
        auto t1 = std::chrono::steady_clock::now();

        omp_set_num_threads(8);
        auto p_cnt = count_nqueens_parallel(n);
        auto t2 = std::chrono::steady_clock::now();

        std::cout << "serial:   " << s_cnt << "  (" << ms_between(t0, t1) << " ms)\n";
        std::cout << "parallel: " << p_cnt << "  (" << ms_between(t1, t2) << " ms)\n";
        std::cout << "expected: " << expected << "\n";
        bool ok = (s_cnt == expected) && (p_cnt == expected);
        std::cout << (ok ? "OK\n" : "MISMATCH\n");
        return ok ? 0 : 2;
    }

    if (mode == "-p") {
        // Prepare output path and file
        const std::string out_dir  = "../output";
        const std::string out_path = out_dir + "/results.csv";
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec); // ok if it already exists

        std::ofstream csv(out_path);
        if (!csv) {
            std::cerr << "Error: could not open " << out_path << " for writing.\n";
            return 1;
        }

        // nice formatting for milliseconds
        csv << std::fixed << std::setprecision(3);

        std::vector<int> Ns  = {8, 10, 12, 14, 15};
        std::vector<int> THs = {1, 2, 4, 8, 16, 32};

        // CSV header (includes total count)
        csv << "n,count,serial_ms";
        for (int th : THs) csv << ",t" << th << "_ms";
        for (int th : THs) if (th > 1) csv << ",speedup_x" << th;
        csv << "\n";

        for (int n : Ns) {
            // serial timing + count (source of truth for 'count' column)
            auto s0 = std::chrono::steady_clock::now();
            std::uint64_t count = count_nqueens_serial(n);
            auto s1 = std::chrono::steady_clock::now();
            double serial_ms = ms_between(s0, s1);

            csv << n << "," << count << "," << serial_ms;

            // parallel timings at different thread counts
            std::vector<double> tpar(THs.size(), 0.0);
            for (size_t i = 0; i < THs.size(); ++i) {
                omp_set_num_threads(THs[i]);
                auto p0 = std::chrono::steady_clock::now();
                std::uint64_t pcnt = count_nqueens_parallel(n);
                auto p1 = std::chrono::steady_clock::now();
                tpar[i] = ms_between(p0, p1);
                csv << "," << tpar[i];

                // sanity check: warn if students' parallelization breaks correctness
                if (pcnt != count) {
                    std::cerr << "Warning: n=" << n << " threads=" << THs[i]
                              << " parallel count=" << pcnt << " != serial count=" << count << "\n";
                }
            }

            // speedups vs serial (threads > 1)
            for (size_t i = 0; i < THs.size(); ++i) {
                if (THs[i] > 1) {
                    double sp = (tpar[i] > 0.0) ? (serial_ms / tpar[i]) : 0.0;
                    csv << "," << sp;
                }
            }
            csv << "\n";
        }

        csv.close();
        std::cout << "Wrote CSV to " << out_path << "\n";
        return 0;
    }

    usage(argv[0]);
    return 1;
}
