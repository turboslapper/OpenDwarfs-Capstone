#include "gol_parallel.h"

// OpenACC version
// Computes the given number of iterations for an n x m array of cells using
// the given number of threads.
void gol_parallel(bool* cells, int n, int m, int iterations, int nthreads) {
    
    int t, i, j;
    int neighbors;
    bool* cells_new = static_cast<bool*>(malloc(n * m * sizeof(bool)));
    bool* temp;

    // OpenACC directive to copy data to GPU once at the start
    #pragma acc data copy(cells[0:n*m]) create(cells_new[0:n*m])
    {
        for(t = 0; t < iterations; ++t) {        
            // OpenACC directive to parallelize the nested loops
            #pragma acc parallel loop collapse(2) present(cells, cells_new)
            for(i = 0; i < n; ++i) {
                for(j = 0; j < m; ++j) {
                    
                    neighbors = 0;
                    if(i>0 && j>0){
                        neighbors += cells[(i-1) * m + j-1] ? 1 : 0;
                        if(i>1 && j>1)
                            neighbors += cells[(i-2) * m + j-2] ? 1 : 0;
                    }
                    if(i>0 && j<m-1){
                        neighbors += cells[(i-1) * m + j+1] ? 1 : 0;
                        if(i>1 && j<m-2)
                            neighbors += cells[(i-2) * m + j+2] ? 1 : 0;
                    }
                    if(i<n-1 && j>0){
                        neighbors += cells[(i+1) * m + j-1] ? 1 : 0;
                        if(i<n-2 && j>1)
                            neighbors += cells[(i+2) * m + j-2] ? 1 : 0;
                    }
                    if(i<n-1 && j<m-1){
                        neighbors += cells[(i+1) * m + j+1] ? 1 : 0;
                        if(i<n-2 && j<m-2)
                            neighbors += cells[(i+2)*m + (j+2)] ? 1 : 0;
                    }

                    cells_new[i * m + j] = neighbors == 2 || (cells[i*m+j] && neighbors == 3);
                }
            }

            temp = cells;
            cells = cells_new;
            cells_new = temp;
        }

        if(iterations % 2) {
            // OpenACC directive to copy data back to CPU
            #pragma acc parallel loop collapse(2) present(cells, cells_new)
            for(i = 0; i < n; ++i) {
                for(j = 0; j < m; ++j) {
                    cells_new[i * m + j] = cells[i * m + j];
                }
            }
        }
    } 

    free(iterations % 2 ? cells : cells_new);
}


// OpenMP version
/*
#include "gol_parallel.h"

// Computes the given number of iterations for an n x m array of cells using
// the given number of threads.
void gol_parallel(bool* cells, int n, int m, int iterations, int nthreads) {
    
    int t, i, j;
    int neighbors;
    bool* cells_new = static_cast<bool*>(malloc(n * m * sizeof(bool)));
    bool* temp;

    for(t = 0; t < iterations; ++t) {
        #pragma omp parallel for schedule(static) num_threads(nthreads) private(j, neighbors)
        for(i = 0; i < n; ++i) {
            for(j = 0; j < m; ++j) {
                
                neighbors = 0;
                if(i>0 && j>0){
                    neighbors += cells[(i-1) * m + j-1] ? 1 : 0;
                    if(i>1 && j>1)
                        neighbors += cells[(i-2) * m + j-2] ? 1 : 0;
                }
                if(i>0 && j<m-1){
                    neighbors += cells[(i-1) * m + j+1] ? 1 : 0;
                    if(i>1 && j<m-2)
                        neighbors += cells[(i-2) * m + j+2] ? 1 : 0;
                }
                if(i<n-1 && j>0){
                    neighbors += cells[(i+1) * m + j-1] ? 1 : 0;
                    if(i<n-2 && j>1)
                        neighbors += cells[(i+2) * m + j-2] ? 1 : 0;
                }
                if(i<n-1 && j<m-1){
                    neighbors += cells[(i+1) * m + j+1] ? 1 : 0;
                    if(i<n-2 && j<m-2)
                        neighbors += cells[(i+2)*m + (j+2)] ? 1 : 0;
                }

                cells_new[i * m + j] = neighbors == 2 || (cells[i*m+j] && neighbors == 3);
            }
        }

        temp = cells;
        cells = cells_new;
        cells_new = temp;
    }

    if(iterations % 2) {
        #pragma omp parallel for schedule(static) num_threads(nthreads) private(j)
        for(i = 0; i < n; ++i) {
            for(j = 0; j < m; ++j) {
                cells_new[i * m + j] = cells[i * m + j];
            }
        }
    }

    free(iterations % 2 ? cells : cells_new);
}
*/
