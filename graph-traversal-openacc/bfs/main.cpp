// main.cpp
// Flags:
//   -p : profile all graphs at ../data/<name>/<name>.mtx using bfs_parallel
//        threads = 1,2,4,8,16,32; writes ../output/bfs_profile.csv with speedups.
//   -c : correctness on hollywood-2009 comparing bfs_parallel vs bfs_serial,
//        also runs threads = 1,2,8,16 and prints timings.
//
// Notes:
// - Measures *only* BFS time (kernel-equivalent), not I/O/build.
// - CSV columns:
//   graph,num_nodes,num_edges,avg_degree,
//   ms_t1,ms_t2,ms_t4,ms_t8,ms_t16,ms_t32,
//   mteps_t1,mteps_t2,mteps_t4,mteps_t8,mteps_t16,mteps_t32,
//   speedup_t2,speedup_t4,speedup_t8,speedup_t16,speedup_t32

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

namespace fs = std::filesystem;

// ---- BFS prototypes (implemented in bfs_serial.cpp / bfs_parallel.cpp)
double bfs_serial(const int* row_ptr, const int* col_idx, int n, int src, int* cost);
double bfs_parallel(const int* row_ptr, const int* col_idx, int n, int src, int* cost, int nthreads);

// ---- Matrix Market loader (coordinate) -> CSR (directed adjacency)
struct CSR {
  int n = 0;                   // vertices
  std::vector<int> row_ptr;    // size n+1
  std::vector<int> col_idx;    // size m
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

  std::vector<std::pair<int,int>> edges;
  edges.reserve((size_t)NZ * (symmetric ? 2 : 1));

  long long read=0;
  while(read < NZ && std::getline(fin,line)){
    std::string t = trim(line);
    if(t.empty() || t[0]=='%') continue;
    std::istringstream iss(t);
    long long i,j; double val=0.0;
    if(!(iss>>i>>j)) continue; // allow trailing values ignored
    edges.emplace_back((int)i,(int)j);
    if(symmetric && i!=j) edges.emplace_back((int)j,(int)i);
    ++read;
  }

  // enforce 0-based
  int min_idx = INT_MAX;
  for(const auto& e: edges){ if(e.first<min_idx) min_idx=e.first; if(e.second<min_idx) min_idx=e.second; }
  bool zero_based_in_file = (min_idx==0);
  if(!zero_based_in_file){
    for(auto &e: edges){ e.first-=1; e.second-=1; }
  }

  out.n = (int)std::max(M, N);
  std::vector<int> deg(out.n, 0);
  for(const auto &e: edges){
    if(e.first<0 || e.first>=out.n || e.second<0 || e.second>=out.n) continue;
    deg[e.first]++;
  }

  out.row_ptr.assign(out.n+1, 0);
  for(int i=0;i<out.n;i++) out.row_ptr[i+1] = out.row_ptr[i] + deg[i];
  out.col_idx.assign(out.row_ptr.back(), 0);

  std::vector<int> cur = out.row_ptr;
  for(const auto &e: edges){
    int u=e.first, v=e.second;
    if(u<0||u>=out.n||v<0||v>=out.n) continue;
    out.col_idx[cur[u]++] = v;
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
                               int n, long long m,
                               const std::vector<double>& ms){
  bool exists = fs::exists(csv_path);
  std::ofstream fout(csv_path, std::ios::app);
  if(!fout){ std::fprintf(stderr,"cannot open %s for append\n", csv_path.string().c_str()); return; }
  if(!exists){
    fout<<"graph,num_nodes,num_edges,avg_degree,"
           "ms_t1,ms_t2,ms_t4,ms_t8,ms_t16,ms_t32,"
           "mteps_t1,mteps_t2,mteps_t4,mteps_t8,mteps_t16,mteps_t32,"
           "speedup_t2,speedup_t4,speedup_t8,speedup_t16,speedup_t32\n";
  }
  auto ms_at = [&](int idx)->double{ return (idx<(int)ms.size() && ms[idx]>0)? ms[idx] : 0.0; };
  double t1 = ms_at(0);
  double avg_deg = (n>0) ? (double)m / (double)n : 0.0;

  double mteps1  = (t1 >0)? ((double)m/(t1/1000.0))/1e6 : 0.0;
  double mteps2  = (ms_at(1)>0)? ((double)m/(ms_at(1)/1000.0))/1e6 : 0.0;
  double mteps4  = (ms_at(2)>0)? ((double)m/(ms_at(2)/1000.0))/1e6 : 0.0;
  double mteps8  = (ms_at(3)>0)? ((double)m/(ms_at(3)/1000.0))/1e6 : 0.0;
  double mteps16 = (ms_at(4)>0)? ((double)m/(ms_at(4)/1000.0))/1e6 : 0.0;
  double mteps32 = (ms_at(5)>0)? ((double)m/(ms_at(5)/1000.0))/1e6 : 0.0;

  auto sp = [&](int idx)->double{ double t=ms_at(idx); return (t1>0 && t>0)? (t1/t) : 0.0; };

  fout<<name<<","<<n<<","<<m<<","<<avg_deg<<","
      <<ms_at(0)<<","<<ms_at(1)<<","<<ms_at(2)<<","<<ms_at(3)<<","<<ms_at(4)<<","<<ms_at(5)<<","
      <<mteps1<<","<<mteps2<<","<<mteps4<<","<<mteps8<<","<<mteps16<<","<<mteps32<<","
      <<sp(1)<<","<<sp(2)<<","<<sp(3)<<","<<sp(4)<<","<<sp(5)<<"\n";
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
    fs::path mtx = fs::path("..") / "data" / "roadNet-CA" / "roadNet-CA.mtx";
    //fs::path mtx = fs::path("..") / "data" / "hollywood-2009" / "hollywood-2009.mtx";
    if(!fs::exists(mtx)){
      std::fprintf(stderr,"hollywood-2009 not found at %s\n", mtx.string().c_str());
      return 1;
    }
    CSR G;
    if(!load_mtx_to_csr(mtx.string(), G)) return 1;

    std::vector<int> cost_ref(G.n, -1);
    double ms_ref = bfs_serial(G.row_ptr.data(), G.col_idx.data(), G.n, 0, cost_ref.data());
    std::printf("[-c] serial   : %s  n=%d  m=%zu  ms=%.3f\n",
                mtx.string().c_str(), G.n, G.col_idx.size(), ms_ref);

    int thread_list[] = {1,2,8,16};
    for(int nt : thread_list){
      std::vector<int> cost_par(G.n, -1);
      double ms_par = bfs_parallel(G.row_ptr.data(), G.col_idx.data(), G.n, 0, cost_par.data(), nt);

      bool ok = (cost_par.size()==cost_ref.size());
      if(ok){
        for(int i=0;i<G.n;i++){ if(cost_par[i]!=cost_ref[i]){ ok=false; break; } }
      }
      std::printf("[-c] parallel : threads=%d  ms=%.3f  %s\n",
                  nt, ms_par, ok ? "OK" : "MISMATCH");
    }
  }

  // -p: profile all graphs with thread sets 1,2,4,8,16,32 -> CSV
  if(do_profile){
    auto files = find_all_graphs_under_data();
    if(files.empty()){
      std::fprintf(stderr,"no graphs found under ../data/<name>/<name>.mtx\n");
      return 1;
    }
    fs::path csv_path = outdir / "bfs_profile.csv";
    for(const auto& mtx : files){
      std::string name = mtx.parent_path().filename().string(); // <name>
      CSR G;
      if(!load_mtx_to_csr(mtx.string(), G)) continue;

      // thread configs and timings
      int thread_list[] = {1,2,4,8,16,32};
      std::vector<double> ms; ms.reserve(6);
      for(int nt : thread_list){
        std::vector<int> cost(G.n, -1);
        double t = bfs_parallel(G.row_ptr.data(), G.col_idx.data(), G.n, 0, cost.data(), nt);
        ms.push_back(t);
      }
      long long m = (long long)G.col_idx.size();
      append_csv_row_ext(csv_path, name, G.n, m, ms);
      std::printf("[-p] %s : n=%d m=%lld  t1=%.3fms t2=%.3f t4=%.3f t8=%.3f t16=%.3f t32=%.3f\n",
                  name.c_str(), G.n, m,
                  ms.size()>0?ms[0]:0, ms.size()>1?ms[1]:0, ms.size()>2?ms[2]:0,
                  ms.size()>3?ms[3]:0, ms.size()>4?ms[4]:0, ms.size()>5?ms[5]:0);
    }
    std::printf("CSV -> %s\n", (outdir / "bfs_profile.csv").string().c_str());
  }

  return 0;
}

