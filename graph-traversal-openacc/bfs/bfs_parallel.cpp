// bfs_parallel.cpp — OpenACC BFS (NVHPC)
// Correct signature for main.cpp.
// Uses raw pointers (vector.data()) so OpenACC atomics are valid.

#include <vector>
#include <chrono>
#include <openacc.h>

double bfs_parallel(const int* row_ptr, const int* col_idx,
                    int n, int src, int* cost, int nthreads)
{
  if(n <= 0 || !row_ptr || !col_idx || !cost) return 0.0;

  // Host init (not timed)
  for(int i = 0; i < n; ++i) cost[i] = -1;
  if(src < 0 || src >= n) return 0.0;

  std::vector<int> graph_mask_v(n, 0);
  std::vector<int> updating_mask_v(n, 0);
  std::vector<int> visited_v(n, 0);

  // IMPORTANT: use raw pointers for device code + atomics
  int* graph_mask    = graph_mask_v.data();
  int* updating_mask = updating_mask_v.data();
  int* visited       = visited_v.data();

  graph_mask[src] = 1;
  visited[src]    = 1;
  cost[src]       = 0;

  const int m = row_ptr[n];   // number of edgess
  if(m <= 0) return 0.0;

  using clock = std::chrono::high_resolution_clock;
  double ms = 0.0;

  #pragma acc data copyin(row_ptr[0:n+1], col_idx[0:m]) \
                   copy(cost[0:n], graph_mask[0:n], updating_mask[0:n], visited[0:n])
  {
    auto t0 = clock::now();

    int over = 0;
    do {
      over = 0;

      // -------------------------
      // Kernel 1: expand frontier
      // -------------------------
      if(nthreads > 0){
        #pragma acc parallel loop gang num_gangs(nthreads)
        for(int tid = 0; tid < n; ++tid){
          if(graph_mask[tid]){
            graph_mask[tid] = 0;

            const int start = row_ptr[tid];
            const int end   = row_ptr[tid + 1];

            #pragma acc loop seq
            for(int e = start; e < end; ++e){
              const int id = col_idx[e];

              int was_visited = 0;
              #pragma acc atomic read
              was_visited = visited[id];

              if(!was_visited){
                // Mark visited (atomic write prevents a true data race)
                #pragma acc atomic write
                visited[id] = 1;

                // For a given BFS level, cost[tid] is the same across the frontier,
                // so redundant writes are harmless (idempotent).
                cost[id] = cost[tid] + 1;
                updating_mask[id] = 1;
              }
            }
          }
        }
      } else {
        #pragma acc parallel loop gang
        for(int tid = 0; tid < n; ++tid){
          if(graph_mask[tid]){
            graph_mask[tid] = 0;

            const int start = row_ptr[tid];
            const int end   = row_ptr[tid + 1];

            #pragma acc loop seq
            for(int e = start; e < end; ++e){
              const int id = col_idx[e];

              int was_visited = 0;
              #pragma acc atomic read
              was_visited = visited[id];

              if(!was_visited){
                #pragma acc atomic write
                visited[id] = 1;

                cost[id] = cost[tid] + 1;
                updating_mask[id] = 1;
              }
            }
          }
        }
      }

      #pragma acc wait

      // -------------------------
      // Kernel 2: build next frontier
      // -------------------------
      if(nthreads > 0){
        #pragma acc parallel loop gang num_gangs(nthreads) reduction(max:over)
        for(int tid = 0; tid < n; ++tid){
          if(updating_mask[tid]){
            graph_mask[tid] = 1;
            updating_mask[tid] = 0;
            over = 1;
          }
        }
      } else {
        #pragma acc parallel loop gang reduction(max:over)
        for(int tid = 0; tid < n; ++tid){
          if(updating_mask[tid]){
            graph_mask[tid] = 1;
            updating_mask[tid] = 0;
            over = 1;
          }
        }
      }

      #pragma acc wait

    } while(over);

    auto t1 = clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  return ms;
}
