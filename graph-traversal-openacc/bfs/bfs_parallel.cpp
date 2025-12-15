// bfs_parallel.cpp
// Race-free level-synchronous BFS using OpenMP.
// Parallelizes expansion of the current frontier.
// Uses atomic CAS on cost[] (initialized to -1) to mark first visit.

#include <vector>
#include <chrono>
#include <omp.h>

double bfs_parallel(const int* row_ptr, const int* col_idx,
                    int n, int src, int* cost, int nthreads)
{
  if(n <= 0 || !row_ptr || !col_idx || !cost) return 0.0;
  if(src < 0 || src >= n) return 0.0;

  // Respect requested thread count if positive; otherwise use OpenMP default.
  if(nthreads > 0) omp_set_num_threads(nthreads);

  // Init (not timed, to match your earlier style)
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < n; ++i) cost[i] = -1;

  cost[src] = 0;

  std::vector<int> frontier;
  frontier.reserve(1024);
  frontier.push_back(src);

  using clock = std::chrono::high_resolution_clock;
  auto t0 = clock::now();

  int level = 0;
  while(!frontier.empty()){
    // Thread-local next frontiers to avoid contention
    int T = omp_get_max_threads();
    std::vector<std::vector<int>> local_next(T);

    // (Optional) reserve some space to reduce reallocs
    // (Heuristic: each thread might discover some nodes)
    for(int t = 0; t < T; ++t) local_next[t].reserve(256);

    // Expand current frontier in parallel
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto &buf = local_next[tid];

      #pragma omp for schedule(dynamic, 64)
      for(int idx = 0; idx < (int)frontier.size(); ++idx){
        int u = frontier[idx];
        int start = row_ptr[u];
        int end   = row_ptr[u + 1];

        for(int e = start; e < end; ++e){
          int v = col_idx[e];

          // Claim v if unvisited: cost[v] == -1 -> set to level+1
          // GCC/Clang builtin CAS is atomic and works well with OpenMP.
          if(__sync_bool_compare_and_swap(&cost[v], -1, level + 1)){
            buf.push_back(v);
          }
        }
      }
    }

    // Merge local buffers (sequential merge is usually fine vs edge traversal cost)
    size_t total = 0;
    for(const auto &buf : local_next) total += buf.size();

    std::vector<int> next_frontier;
    next_frontier.reserve(total);
    for(auto &buf : local_next){
      next_frontier.insert(next_frontier.end(), buf.begin(), buf.end());
    }

    frontier.swap(next_frontier);
    ++level;
  }

  auto t1 = clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return ms;
}
