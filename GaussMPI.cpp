#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#define MAXN 2000// Maximum Dimension for Matrix 

void initializeMat();
void backSubstitution();
void displayMat();
void printAnswer();
void gauss(int N);

int proc, id, N;
/* Matrices and vectors */
float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

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

/* Displays the matrix which has being initialized */
void displayMat()
{
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

/* This function performs the backsubstitution */
void backSubstitution()
{
	int row, col;
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];
		for (col = N - 1; col > row; col--) {
			X[row] -= A[row][col] * X[col];
		}
		X[row] /= A[row][row];
	}
}

/* This function performs gaussian elimination with MPI implementation through static interleave */
void gauss()
{
	double wp_time, wa_time = 0;
	MPI_Request request;
	MPI_Status status;
	int p, k, i, j;
	float mp;

	MPI_Barrier(MPI_COMM_WORLD);// waiting for all processors	
	if (id == 0)// Processors starts the MPI Timer i.e MPI_Wtime()
	{
		wa_time = MPI_Wtime();
	}

	for (k = 0; k < N - 1; k++)
	{
		//Broadcsting A's and B's matrix from 0th rank processor to all other processors.
		MPI_Bcast(&A[k][0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&B[k], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if (id == 0)
		{
			for (p = 1; p < proc; p++)
			{
				for (i = k + 1 + p; i < N; i += proc)
				{
					/* Sending A and B matrix from oth to all other processors using non blocking send*/
					MPI_Isend(&A[i], N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					MPI_Isend(&B[i], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
				}
			}
			// implementing gaussian elimination 
			for (i = k + 1; i < N; i += proc)
			{
				mp = A[i][k] / A[k][k];
				for (j = k; j < N; j++)
				{
					A[i][j] -= A[k][j] * mp;
				}
				B[i] -= B[k] * mp;
			}
			// Receiving all the values that are send by 0th processor.
			for (p = 1; p < proc; p++)
			{
				for (i = k + 1 + p; i < N; i += proc)
				{
					MPI_Recv(&A[i], N, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
					MPI_Recv(&B[i], 1, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
				}
			}
			//Stopping the MPI_Timer
			if (k == N - 2)
			{
				wp_time = MPI_Wtime();
				printf("Time for calculating = %f\n", wp_time - wa_time);
			}
		}
		else
		{
			for (i = k + 1 + id; i < N; i += proc)
			{
				MPI_Recv(&A[i], N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&B[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				mp = A[i][k] / A[k][k];
				for (j = k; j < N; j++)
				{
					A[i][j] -= A[k][j] * mp;
				}
				B[i] -= B[k] * mp;
				MPI_Isend(&A[i], N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				MPI_Isend(&B[i], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);//Waiting for all processors
	}
}

void printAnswer()
{
	int row;

	if (N < 10) {
		printf("\nX = [");
		for (row = 0; row < N; row++) {
			printf("%5.2f%s", X[row], (row < N - 1) ? "; " : "]\n");
		}
	}
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);//Initiating MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &id);//Getting rank of current processor.
	MPI_Comm_size(MPI_COMM_WORLD, &proc);//Getting number of processor in MPI_COMM_WORLD

	N = 2000;
	printf("\nMatrix dimension N = %i.\n", N);

	unsigned int start_time;
	if (id == 0)
	{
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
		backSubstitution();

		/* Stop Clock */
		unsigned int end_time = clock();
		printf("Stopped clock.\n");

		/* Display output */
		printAnswer();

		unsigned int search_time = end_time - start_time;
		//printf("Time for calculating: %i ms)\n", search_time);
	}
	MPI_Finalize(); //Finalizing the MPI
	return 0;
}
