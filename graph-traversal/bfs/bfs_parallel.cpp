// bfs_parallel.cpp
// Basic OpenMP parallelization - has room for optimization!
#include <vector>
#include <chrono>
#include <omp.h>

double bfs_parallel(const int* row_ptr, const int* col_idx, int n, int src, int* cost, int nthreads){
  if(n<=0 || !row_ptr || !col_idx || !cost) return 0.0;

  std::vector<int> graph_mask(n,0);
  std::vector<int> updating_mask(n,0);
  std::vector<int> visited(n,0);

  for(int i=0;i<n;i++) cost[i] = -1;
  if(src<0 || src>=n) return 0.0;

  graph_mask[src]=1;
  visited[src]=1;
  cost[src]=0;

  omp_set_num_threads(nthreads);

  using clock = std::chrono::high_resolution_clock;
  auto t0 = clock::now();

  int over;
  do{
    over = 0;

    // kernel1 - parallelize outer loop
    #pragma omp parallel for
    for(int tid=0; tid<n; ++tid){
      if(graph_mask[tid]!=0){
        graph_mask[tid]=0;
        int start = row_ptr[tid];
        int end   = row_ptr[tid+1];
        for(int i=start; i<end; ++i){
          int id = col_idx[i];
          if(!visited[id]){
            cost[id] = cost[tid] + 1;
            updating_mask[id] = 1;
          }
        }
      }
    }

    // kernel2 - parallelize outer loop
    #pragma omp parallel for
    for(int tid=0; tid<n; ++tid){
      if(updating_mask[tid]==1){
        graph_mask[tid]=1;
        visited[tid]=1;
        over=1;
        updating_mask[tid]=0;
      }
    }
  } while(over);

  auto t1 = clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return ms;
}

