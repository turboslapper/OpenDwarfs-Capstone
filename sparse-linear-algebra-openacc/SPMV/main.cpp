// main.cpp — SPMV driver (reads ../data/<graph>/<graph>.mtx, runs serial/parallel SPMV)
// Flags:
//   -c : correctness on hollywood-2009; runs threads = 1,2,8,16 and checks vs serial
//   -p : profile ALL graphs under ../data; threads = 1,2,4,8,16,32; writes ../output/spmv_profile.csv (no MTEPS)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <climits>
#include <cmath>

namespace fs = std::filesystem;

// ---- SPMV prototypes (implemented in spmv_serial.cpp / spmv_parallel.cpp)
double spmv_serial(const int* row_ptr, const int* col_idx, const double* val,
                   int nrows, const double* x, double* y);
double spmv_parallel(const int* row_ptr, const int* col_idx, const double* val,
                     int nrows, const double* x, double* y, int nthreads);

// ---- Matrix Market loader (coordinate) -> CSR with values
struct CSR {
  int nrows = 0;
  int ncols = 0;
  std::vector<int> row_ptr;   // size nrows+1
  std::vector<int> col_idx;   // size nnz
  std::vector<double> val;    // size nnz
};

static inline std::string trim(const std::string& s){
  size_t a=0,b=s.size();
  while(a<b && std::isspace((unsigned char)s[a])) ++a;
  while(b>a && std::isspace((unsigned char)s[b-1])) --b;
  return s.substr(a,b-a);
}

static bool load_mtx_to_csr(const std::string& path, CSR& out){
  std::ifstream fin(path);
  if(!fin){ std::fprintf(stderr,"cannot open %s\n", path.c_str()); return false; }

  std::string line;
  if(!std::getline(fin, line)){ std::fprintf(stderr,"empty file %s\n", path.c_str()); return false; }

  std::string hdr = trim(line);
  std::string lower = hdr;
  for(char& c: lower) c = (char)std::tolower((unsigned char)c);
  bool symmetric = (lower.find("symmetric") != std::string::npos);

  long long M=0,N=0,NZ=0;
  // skip comments until size line
  while(std::getline(fin,line)){
    std::string t = trim(line);
    if(t.empty() || t[0]=='%') continue;
    std::istringstream iss(t);
    if(!(iss>>M>>N>>NZ)){ std::fprintf(stderr,"bad size line in %s\n", path.c_str()); return false; }
    break;
  }
  if(M<=0 || N<=0 || NZ<0){ std::fprintf(stderr,"invalid dims in %s\n", path.c_str()); return false; }

  struct Trip { int i,j; double v; };
  std::vector<Trip> trips;
  trips.reserve((size_t)NZ * (symmetric ? 2 : 1));

  long long read=0;
  while(read < NZ && std::getline(fin,line)){
    std::string t = trim(line);
    if(t.empty() || t[0]=='%') continue;
    std::istringstream iss(t);
    long long i,j; double v=1.0;
    if(!(iss>>i>>j)) continue;
    if(!(iss>>v)) v = 1.0; // pattern/int without explicit value -> 1.0
    trips.push_back({(int)i,(int)j,v});
    if(symmetric && i!=j) trips.push_back({(int)j,(int)i,v});
    ++read;
  }

  // enforce 0-based indexing
  int min_idx = INT_MAX;
  for(const auto& e: trips){ if(e.i<min_idx) min_idx=e.i; if(e.j<min_idx) min_idx=e.j; }
  if(min_idx != 0){
    for(auto &e: trips){ e.i -= 1; e.j -= 1; }
  }

  out.nrows = (int)M;
  out.ncols = (int)N;

  std::vector<int> deg(out.nrows, 0);
  for(const auto &e: trips){
    if(e.i<0 || e.i>=out.nrows || e.j<0 || e.j>=out.ncols) continue;
    deg[e.i]++;
  }

  out.row_ptr.assign(out.nrows+1, 0);
  for(int r=0;r<out.nrows;r++) out.row_ptr[r+1] = out.row_ptr[r] + deg[r];
  out.col_idx.assign(out.row_ptr.back(), 0);
  out.val.assign(out.row_ptr.back(), 0.0);

  std::vector<int> cur = out.row_ptr;
  for(const auto &e: trips){
    int r=e.i, c=e.j;
    if(r<0||r>=out.nrows||c<0||c>=out.ncols) continue;
    int pos = cur[r]++;
    out.col_idx[pos] = c;
    out.val[pos]     = e.v;
  }
  return true;
}

// ---- helpers: data discovery, I/O
static std::vector<fs::path> find_all_graphs_under_data(){
  std::vector<fs::path> files;
  fs::path data = fs::path("..") / "data";
  std::error_code ec;
  if(!fs::exists(data, ec)) return files;
  for(auto& entry : fs::directory_iterator(data, ec)){
    if(!entry.is_directory()) continue;
    std::string base = entry.path().filename().string();
    fs::path mtx = entry.path() / (base + ".mtx");
    if(fs::exists(mtx)) files.push_back(mtx);
  }
  std::sort(files.begin(), files.end());
  return files;
}

static bool ensure_dir(const fs::path& p){
  std::error_code ec;
  if(fs::exists(p, ec)) return true;
  return fs::create_directories(p, ec);
}

static void append_csv_row_ext(const fs::path& csv_path,
                               const std::string& name,
                               int nrows, int ncols, long long nnz,
                               const std::vector<double>& ms){
  bool exists = fs::exists(csv_path);
  std::ofstream fout(csv_path, std::ios::app);
  if(!fout){ std::fprintf(stderr,"cannot open %s for append\n", csv_path.string().c_str()); return; }
  if(!exists){
    fout<<"graph,num_rows,num_cols,nnz,avg_degree,"
           "ms_t1,ms_t2,ms_t4,ms_t8,ms_t16,ms_t32,"
           "speedup_t2,speedup_t4,speedup_t8,speedup_t16,speedup_t32\n";
  }
  auto ms_at = [&](int idx)->double{ return (idx<(int)ms.size() && ms[idx]>0)? ms[idx] : 0.0; };
  double t1 = ms_at(0);
  double avg_deg = (nrows>0) ? (double)nnz / (double)nrows : 0.0;
  auto sp = [&](int idx)->double{ double t=ms_at(idx); return (t1>0 && t>0)? (t1/t) : 0.0; };

  fout<<name<<","<<nrows<<","<<ncols<<","<<nnz<<","<<avg_deg<<","
      <<ms_at(0)<<","<<ms_at(1)<<","<<ms_at(2)<<","<<ms_at(3)<<","<<ms_at(4)<<","<<ms_at(5)<<","
      <<sp(1)<<","<<sp(2)<<","<<sp(3)<<","<<sp(4)<<","<<sp(5)<<"\n";
}

static bool vec_equal(const std::vector<double>& a, const std::vector<double>& b, double eps=1e-9){
  if(a.size()!=b.size()) return false;
  for(size_t i=0;i<a.size();++i){
    double diff = std::fabs(a[i]-b[i]);
    if(diff > eps * std::max(1.0, std::fabs(a[i]))) return false;
  }
  return true;
}

// ---- main
int main(int argc, char** argv){
  bool do_profile=false, do_correct=false;
  for(int i=1;i<argc;i++){
    if(std::strcmp(argv[i], "-p")==0 || std::strcmp(argv[i], "-P")==0) do_profile=true;
    else if(std::strcmp(argv[i], "-c")==0) do_correct=true;
  }

  fs::path outdir = fs::path("..") / "output";
  ensure_dir(outdir);

  // -c: correctness & timing on hollywood-2009, threads = 1,2,8,16
  if(do_correct){
    fs::path mtx = fs::path("..") / "data" / "hollywood-2009" / "hollywood-2009.mtx";
    if(!fs::exists(mtx)){
      std::fprintf(stderr,"hollywood-2009 not found at %s\n", mtx.string().c_str());
      return 1;
    }
    CSR A;
    if(!load_mtx_to_csr(mtx.string(), A)) return 1;

    std::vector<double> x(A.ncols, 1.0), y_ref(A.nrows, 0.0);

    double ms_ref = spmv_serial(A.row_ptr.data(), A.col_idx.data(), A.val.data(),
                                A.nrows, x.data(), y_ref.data());
    std::printf("[-c] serial   : %s  nrows=%d  ncols=%d  nnz=%zu  ms=%.3f\n",
                mtx.string().c_str(), A.nrows, A.ncols, A.col_idx.size(), ms_ref);

    int thread_list[] = {1,2,8,16};
    for(int nt : thread_list){
      std::vector<double> y(A.nrows, 0.0);
      double ms_par = spmv_parallel(A.row_ptr.data(), A.col_idx.data(), A.val.data(),
                                    A.nrows, x.data(), y.data(), nt);
      bool ok = vec_equal(y_ref, y, 1e-9);
      std::printf("[-c] parallel : threads=%d  ms=%.3f  %s\n",
                  nt, ms_par, ok ? "OK" : "MISMATCH");
    }
  }

  // -p: profile all graphs with thread sets 1,2,4,8,16,32 -> CSV (no MTEPS)
  if(do_profile){
    auto files = find_all_graphs_under_data();
    if(files.empty()){
      std::fprintf(stderr,"no matrices found under ../data/<name>/<name>.mtx\n");
      return 1;
    }
    fs::path csv_path = outdir / "spmv_profile.csv";
    for(const auto& mtx : files){
      std::string name = mtx.parent_path().filename().string(); // <name>
      CSR A;
      if(!load_mtx_to_csr(mtx.string(), A)) continue;

      std::vector<double> x(A.ncols, 1.0);
      int thread_list[] = {1,2,4,8,16,32};
      std::vector<double> ms; ms.reserve(6);
      for(int nt : thread_list){
        std::vector<double> y(A.nrows, 0.0);
        double t = spmv_parallel(A.row_ptr.data(), A.col_idx.data(), A.val.data(),
                                 A.nrows, x.data(), y.data(), nt);
        ms.push_back(t);
      }
      long long nnz = (long long)A.col_idx.size();
      append_csv_row_ext(csv_path, name, A.nrows, A.ncols, nnz, ms);
      std::printf("[-p] %s : nrows=%d ncols=%d nnz=%lld  t1=%.3fms t2=%.3f t4=%.3f t8=%.3f t16=%.3f t32=%.3f\n",
                  name.c_str(), A.nrows, A.ncols, nnz,
                  ms.size()>0?ms[0]:0, ms.size()>1?ms[1]:0, ms.size()>2?ms[2]:0,
                  ms.size()>3?ms[3]:0, ms.size()>4?ms[4]:0, ms.size()>5?ms[5]:0);
    }
    std::printf("CSV -> %s\n", (outdir / "spmv_profile.csv").string().c_str());
  }

  return 0;
}

