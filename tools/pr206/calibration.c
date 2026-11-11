#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) return 1;
    errno = EDOM;
    void *p = malloc(6817);
    void *a = NULL;
    if (!p || posix_memalign(&a, 256, 6833) || (uintptr_t)a % 256) return 2;
    void *q = realloc(p, 6841);
    void *zeros = calloc(37, 197);
    if (!q || !zeros || ((unsigned char *)zeros)[54] || errno != EDOM) return 3;
    free(q); free(a); free(zeros);
    return 0;
}
