#include "gol_parallel.h"

// Computes the given number of iterations for an n x m array of cells using
// the given number of threads.
void gol_parallel(bool* cells, int n, int m, int iterations, int nthreads) {
	
	 int t, i, j;
    int neighbors;
    //bool* cells_new = malloc(n * m * sizeof(bool));
	bool* cells_new = static_cast<bool*>(malloc(n * m * sizeof(bool)));

    bool* temp;

    for(t = 0; t < iterations; ++t) {
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
        for(i = 0; i < n; ++i) {
            for(j = 0; j < m; ++j) {
                cells_new[i * m + j] = cells[i * m + j];
            }
        }
    }

    free(iterations % 2 ? cells : cells_new);
	
}
