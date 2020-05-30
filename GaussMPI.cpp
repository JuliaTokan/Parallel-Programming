#include<stdio.h>
#include<mpi.h>
#include <stdlib.h>
#include <time.h>

#define MAXN 2000

#define randm() 4|2[uid]&3

double A[MAXN][MAXN + 1], X[MAXN + 1], mp, val;
int proc, id, i, j, v, k, k1, p;
int rt, t1, t2;

void initializeMat() {
    int row, col;

    printf("\nInitializing...\n");
    for (col = 0; col < MAXN; col++) {
        for (row = 0; row < MAXN+1; row++) {
            A[col][row] = (float)rand() / 32768.0+0.1;
        }
        X[col] = 0.0;
    }
}

void displayMat()
{
    int row, col;
    if (MAXN < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < MAXN; row++) {
            for (col = 0; col < MAXN; col++) {
                printf("%5.2f%s", A[row][col], (col < MAXN - 1) ? ", " : ";\n\t");
            }
        }
        printf("\nB = [");
        for (col = 0; col < MAXN; col++) {
            printf("%5.2f%s", A[col][MAXN], (col < MAXN - 1) ? "; " : "]\n");
        }
    }
}

void printAnswer()
{
    int row;

    if (MAXN < 10) {
        printf("\nX = [");
        for (row = 0; row < MAXN; row++) {
            printf("%5.2f%s", A[row][MAXN], (row < MAXN - 1) ? "; " : "]\n");
        }
    }
}

void gauss()
{
    int N = MAXN / proc;

    for (k = 0; k < N; k++)
    {
        for (p = 0; p < proc; p++)
        {
            if (id == p)
            {
                mp = A[k][k] / A[k][proc * k + p];
                for (j = MAXN; j >= proc * k + p; j--)
                    A[k][j] = A[k][j] * mp;
                for (j = 0; j <= MAXN; j++)
                    X[j] = A[k][j];
                MPI_Bcast(X, MAXN + 1, MPI_DOUBLE, p, MPI_COMM_WORLD);
                for (i = k + 1; i < N; i++)
                {
                    for (j = MAXN; j >= proc * k + p; j--)
                        A[i][j] = A[i][j] - A[i][proc * k + p] * A[k][j];
                }
            }
            else if (id < p)
            {
                MPI_Bcast(X, MAXN + 1, MPI_DOUBLE, p, MPI_COMM_WORLD);
                for (i = k + 1; i < N; i++)
                {
                    for (j = MAXN; j >= proc * k + p; j--)
                        A[i][j] = A[i][j] - A[i][proc * k + p] * X[j];
                }
            }
            else if (id > p)
            {
                MPI_Bcast(X, MAXN + 1, MPI_DOUBLE, p, MPI_COMM_WORLD);
                for (i = k; i < N; i++)
                {
                    for (j = MAXN; j >= proc * k + p; j--)
                        A[i][j] = A[i][j] - A[i][proc * k + p] * X[j];
                }
            }
        }
    }


    /* backSubstitution */
    for (k1 = N - 2, k = N - 1; k >= 0; k--, k1--)
    {
        for (p = proc - 1; p >= 0; p--)
        {
            if (id == p)
            {
                val = A[k][MAXN];
                MPI_Bcast(&val, 1, MPI_DOUBLE, p, MPI_COMM_WORLD);
                for (i = k - 1; i >= 0; i--)
                    A[i][MAXN] -= A[k][MAXN] * A[i][proc * k + p];
            }
            else if (id < p)
            {
                MPI_Bcast(&val, 1, MPI_DOUBLE, p, MPI_COMM_WORLD);
                for (i = k; i >= 0; i--)
                    A[i][MAXN] -= val * A[i][proc * k + p];
            }
            else if (id > p)
            {
                MPI_Bcast(&val, 1, MPI_DOUBLE, p, MPI_COMM_WORLD);
                for (i = k1; i >= 0; i--)
                    A[i][MAXN] -= val * A[i][proc * k + p];
            }
        }
    }
}

int main(int args, char** argv)
{
    MPI_Init(&args, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    unsigned int start_time;
    if (id == 0)
    {
        printf("\nMatrix dimension N = %i.\n", MAXN);

        /* Initialize A and B */
        initializeMat();

        /* Print input matrices */
        displayMat();

        /* Start Clock */
        printf("\nStarting clock.\n");
        start_time = clock();
    }

    gauss();//implementing the gaussian elimination

    if (id == 0)
    {
        /* Stop Clock */
        unsigned int end_time = clock();
        printf("Stopped clock.\n");

        /* Display output */
        printAnswer();

        unsigned int search_time = end_time - start_time;
        printf("Time for calculating: %i ms)\n", search_time);
    }

    MPI_Finalize();
    return(0);
}
