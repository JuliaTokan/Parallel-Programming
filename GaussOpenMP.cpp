#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <time.h>
#include <omp.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

void gauss();

/* Initialize A and B (and X to 0.0s) */
void initializeMat() {
    int row, col;

    printf("\nInitializing...\n");
    for (col = 0; col < N; col++) {
        for (row = 0; row < N; row++) {
            A[row][col] = (float)rand() / 32768.0;
        }
        B[col] = (float)rand() / 32768.0;
        X[col] = 0.0;
    }

}

/* Print input matrices */
void displayMat() {
    int row, col;
    if (N < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t");
            }
        }
        printf("\nB = [");
        for (col = 0; col < N; col++) {
            printf("%5.2f%s", B[col], (col < N - 1) ? "; " : "]\n");
        }
    }
}

void printAnswer() {
    int row;
    if (N < 10) {
        printf("\nX = [");
        for (row = 0; row < N; row++) {
            printf("%5.2f%s", X[row], (row < N - 1) ? "; " : "]\n");
        }
    }
}

int main(int argc, char** argv) {
    /* Process program parameters */
    N = 2000;
    printf("\nMatrix dimension N = %i.\n", N);

    /* Initialize A and B */
    initializeMat();

    /* Print input matrices */
    displayMat();

    /* Start Clock */
    printf("\nStarting clock.\n");
    unsigned int start_time = clock();

    /* Gaussian Elimination */
    gauss();

    /* Stop Clock */
    unsigned int end_time = clock();
    printf("Stopped clock.\n");
 
    /* Display output */
    printAnswer();

    unsigned int search_time = end_time - start_time;
    printf("Time for calculating: %i ms)\n",
        search_time);

    exit(0);
}

void gauss() {
    int norm, row, col;  /* Normalization row, and zeroing
              * element row and col */
    float multiplier;

    /* Gaussian elimination */
    for (norm = 0; norm < N - 1; norm++) {
#pragma omp parallel for shared(A, B) private(multiplier,row,col)
        for (row = norm + 1; row < N; row++) {
            multiplier = A[row][norm] / A[norm][norm];
            for (col = norm; col < N; col++) {
                A[row][col] -= A[norm][col] * multiplier;
            }
            B[row] -= B[norm] * multiplier;
        }
    }
    /* (Diagonal elements are not normalized to 1.  This is treated in back
     * substitution.)
     */


     /* Back substitution */
    for (row = N - 1; row >= 0; row--) {
        X[row] = B[row];
        for (col = N - 1; col > row; col--) {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
}