#include "gameoflife.h"
#include <pthread.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
void gol_parallel(bool*, int, int, int, int);
#ifdef __cplusplus
}
#endif
