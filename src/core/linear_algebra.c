//Linear algebra class of LibBQsC by Berkay

#include "../../include/core/linear_algebra.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Vector methods of the linear algebra class
 */

//The method to initialize a vector of zeros
double* initVector(int n)
{
	//Allocate the memory for the vector
	double* vector = (double*) malloc (n * sizeof(double));
	//Handle any allocation failure
	if (vector == NULL)
	{
		exit(EXIT_FAILURE);
	}
	//Initialize the array with zeros
	for (int i = 0; i < n; i++)
	{
		vector[i] = 0.0;
	}
	//Return the vector
	return vector;
}

//Method to initialize a vector of zeros - separate method for clarity
double* initZeroVector(int n)
{
	//Allocate the memory for the vector
	double* vector = (double*) malloc (n * sizeof(double));
	//Handle any allocation failure
	if (vector == NULL)
	{
		exit(EXIT_FAILURE);
	}
	//Initialize the array with zeros
	for (int i = 0; i < n; i++)
	{
		vector[i] = 0.0;
	}
	//Return the vector
	return vector;
}

//Method to initialize a vector of random numbers
double* initRandomVector(int n)
{
	//Allocate the memory for the vector
	double* vector = (double*) malloc (n * sizeof(double));
	//Handle any allocation failure
	if (vector == NULL)
	{
		exit(EXIT_FAILURE);
	}
	//Seed the random number generator
	srand(time(NULL));
	//Initialize the array with zeros
	for (int i = 0; i < n; i++)
	{
		//Define the random number
		vector[i] = (double)rand() / RAND_MAX;
	}
	//Return the vector
	return vector;
}

//The method to append a number to a vector
double* vectorAppend(double* vector, int n, double number, int index, int dispose)
{
	//Allocate the new memory for the vector
	double* new_vector = (double*) malloc ((n + 1) * sizeof(double));
	//Handle any allocation failure
	if (new_vector == NULL)
	{
		exit(EXIT_FAILURE);
	}
	//Add the numbers to the new vector
	for (int i = 0; i < (n + 1); i++)
	{
		if (i < index)
		{
			new_vector[i] = vector[i];
		}
		else if (i == index)
		{
			new_vector[i] = number;
		}
		else if (i > index)
		{
			new_vector[i] = vector[i-1];
		}
	}
	//Dispose the old vector
	if (dispose == 1)
	{
		free(vector);
	}
	//Return the new vector
	return new_vector;
}

//The method for vector addition
double* vectorAddition(double* a, double* b, int n)
{
	//Initialize the resulting vector
	double* result = initVector(n);
	//Do the vector addition into the result vector
	for (int i = 0; i < n; i++)
	{
		result[i] = a[i] + b[i];
	}
	//Return the result vector
	return result;
}

//The method for vector subtraction
double* vectorSubtraction(double* a, double* b, int n)
{
	//Initialize the resulting vector
	double* result = initVector(n);
	//Do the vector subtraction into the result vector
	for (int i = 0; i < n; i++)
	{
		result[i] = a[i] - b[i];
	}
	//Return the result vector
	return result;
}

//The method for scalar multiplication
double* vectorScalarMultiplication(double* vector, int n, double scalar)
{
	//Initialize the resulting vector
	double* result = initVector(n);
	//Do the scalar multiplication into the result vector
	for (int i = 0; i < n; i++)
	{
		result[i] = scalar * vector[i];
	}
	//Return the result vector
	return result;
}

//The method dot product
double vectorDotProduct(double* a, double* b, int n)
{
	double result = 0;
	//Calculate the result
	for (int i = 0; i < n; i++)
	{
		result += a[i] * b[i];
	}
	//Return the result
	return result;
}

//Method to print a vector
void printVector(double* a, int n, int decimal_places)
{
	printf("[");
	//Print the elements on the vector
	for (int i = 0; i < n; i++)
	{
		printf(" %.*f", decimal_places, a[i]);
	}
	//Print the brace and \n
	printf(" ]\n");
}

/**
 * Matrix methods of the linear algebra class
 */

//Method for generating double**
double** initMatrix(int rows, int columns)
{
	//Initialize the double** array
	double** array = (double**) malloc (rows * sizeof(double*));
	//Handle the allocation failure for the array
	if (array == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Define the columns of the array
	for (int i = 0; i < rows; i++)
	{
		array[i] = (double*) malloc (columns * sizeof(double));
		//Handle the allocation failure for the current row of the array
		if (array[i] == NULL)
		{
			printf("Failed to allocate memory");
			exit(EXIT_FAILURE);
		}
	}
	//Return the array
	return array;
}

//Method for generating double** with zeroes
double** initZeroMatrix(int rows, int columns)
{
	//Initialize the double** array
	double** array = (double**) malloc (rows * sizeof(double*));
	//Handle the allocation failure for the array
	if (array == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Define the columns of the array
	for (int i = 0; i < rows; i++)
	{
		array[i] = (double*) malloc (columns * sizeof(double));
		//Handle the allocation failure for the current row of the array
		if (array[i] == NULL)
		{
			printf("Failed to allocate memory");
			exit(EXIT_FAILURE);
		}
		//Define the items of this row as zeroes
		for (int j = 0; j < columns; j++)
		{
			array[i][j] = 0.0;
		}
	}
	//Return the array
	return array;
}

//Method for generating double** with random number between 0 and 1
double** initRandomMatrix(int rows, int columns)
{
	//Initialize the double** array
	double** array = (double**) malloc (rows * sizeof(double*));
	//Handle the allocation failure for the array
	if (array == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Seed the random number generator
	srand(time(NULL));
	//Define the columns of the array
	for (int i = 0; i < rows; i++)
	{
		array[i] = (double*) malloc (columns * sizeof(double));
		//Handle the allocation failure for the current row of the array
		if (array[i] == NULL)
		{
			printf("Failed to allocate memory");
			exit(EXIT_FAILURE);
		}
		//Define the items of this row as zeroes
		for (int j = 0; j < columns; j++)
		{
			//Define the random number
			array[i][j] = (double)rand() / RAND_MAX;
		}
	}
	//Return the array
	return array;
}

//Method to get a row from a matrix; returns the clone of the row
double* matrixGetRow(double** A, int rows, int columns, int index)
{
	//Initialize the empty vector
	double* row = initVector(columns);
	//Copy the items into the vector
	for (int i = 0; i < columns; i++)
	{
		row[i] = A[index][i];
	}
	//Return the vector
	return row;
}

//Method to get a column from a matrix; returns the clone of the column
double* matrixGetColumn(double** A, int rows, int columns, int index)
{
	//Initialize the empty vector
	double* column = initVector(rows);
	//Copy the items into the vector
	for (int i = 0; i < rows; i++)
	{
		column[i] = A[i][index];
	}
	//Return the vector
	return column;
}

//The method for matrix addition
double** matrixAddition(double** A, double** B, int rows, int columns)
{
	//Generate the empty result matrix
	double** result_matrix = initMatrix(rows, columns);
	//Fill the result matrix accordingly
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			result_matrix[i][j] = A[i][j] + B[i][j];
		}
	}
	//Return the result matrix
	return result_matrix;
}

//The method for matrix subtraction
double** matrixSubtraction(double** A, double** B, int rows, int columns)
{
	//Generate the empty result matrix
	double** result_matrix = initMatrix(rows, columns);
	//Fill the result matrix accordingly
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			result_matrix[i][j] = A[i][j] - B[i][j];
		}
	}
	//Return the result matrix
	return result_matrix;
}

//The method for matrix scalar multiplication
double** matrixScalarMultiplication(double** A, int rows, int columns, double scalar)
{
	//Generate the empty result matrix
	double** result_matrix = initMatrix(rows, columns);
	//Fill the result matrix accordingly
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			result_matrix[i][j] = A[i][j] * scalar;
		}
	}
	//Return the result matrix
	return result_matrix;
}

//The method for matrix multiplication
double** matrixMultiplication(double** A, int rows_A, int columns_A, double** B, int rows_B, int columns_B)
{
	//If the matrices have the required dimensions
	if (columns_A == rows_B)
	{
		//Get the empty result matrix
		double** result_matrix = initMatrix(rows_A, columns_B);
		//Iterate on the rows of A
		for (int a_row_no = 0; a_row_no < rows_A; a_row_no++)
		{
			//Iterate on the columns of B
			for (int b_column_no = 0; b_column_no < columns_B; b_column_no++)
			{
				double new_item = 0;
				//Iterate on the items
				for (int index = 0; index < columns_A; index++)
				{
					new_item += A[a_row_no][index] * B[index][b_column_no];
				}
				result_matrix[a_row_no][b_column_no] = new_item;
			}
		}
		//Return the result matrix
		return result_matrix;
	}
	else
	{
		printf("Invalid dimensions for matrix multiplication");
		exit(EXIT_FAILURE);
	}
}

//The method for transpose
double** matrixTranspose(double** A, int rows, int columns)
{
	//Get the empty result matrix
	double** result_matrix = initMatrix(columns, rows);
	//Fill the result matrix accordingly
	for (int i = 0; i < columns; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			result_matrix[i][j] = A[j][i];
		}
	}
	//Return the result matrix
	return result_matrix;
}

//The method for sub-matrix
double** matrixSubmatrix(double** A, int rows, int columns, int i, int j)
{
	//If the matrix is big enough
	if (rows > 1 && columns > 1)
	{
		//Get the empty result matrix
		double** result_matrix = initMatrix(rows-1, columns-1);
		//Iterate on the rows of the result matrix
		for (int row_no = 0; row_no < rows-1; row_no++)
		{
			//Iterate on the columns of the result matrix
			for (int column_no = 0; column_no < columns-1; column_no++)
			{
				//Indices of the items to be taken from the passed matrix
				int row_index_in_original_matrix = row_no;
				int column_index_in_original_matrix = column_no;
				//Drop the ith row
				if (row_no < i)
				{
					row_index_in_original_matrix = row_no;
				}
				else
				{
					row_index_in_original_matrix = row_no + 1;
				}
				//Drop the jth column
				if (column_no < j)
				{
					column_index_in_original_matrix = column_no;
				}
				else
				{
					column_index_in_original_matrix = column_no + 1;
				}
				//Add the correct item to the result matrix
				result_matrix[row_no][column_no] = A[row_index_in_original_matrix][column_index_in_original_matrix];
			}
		}
		//Return the result matrix
		return result_matrix;
	}
	else
	{
		printf("Sub matrix does not exist for this Matrix");
		exit(EXIT_FAILURE);
	}
}

//The method for determinant using Laplace Expansion
double matrixDeterminantLaplaceExpansion(double** A, int rows, int columns)
{
	//If the matrix is square
	if (rows == columns)
	{
		//If the matrix is 1x1
		if (rows == 1 && columns == 1)
		{
			return A[0][0];
		}
		//Otherwise, do the Laplace expansion with recursion
		else
		{
			double determinant = 0;
			//Iterate in the items on the first row of the matrix
			for (int index = 0; index < columns; index++)
			{
				//Sum of item * sign factor * minor
				int sign_factor = ((0 + index) % 2 == 0) ? 1 : -1;
				double** sub_matrix = matrixSubmatrix(A, rows, columns, 0, index);
				determinant += A[0][index] * sign_factor * matrixDeterminantLaplaceExpansion(sub_matrix, rows-1, columns-1);
				//Dispose the sub matrix
				matrixDispose(sub_matrix, rows-1);
			}
			//Return the determinant
			return determinant;
		}
	}
	else
	{
		printf("Determinant of a non-square Matrix cannot be calculated");
		exit(EXIT_FAILURE);
	}
}

//The method for determinant
double matrixDeterminant(double** A, int rows, int columns)
{
	//There is single method to find a determinant for now, new one will be added
	return matrixDeterminantLaplaceExpansion(A, rows, columns);
}

//The method for a minor of a matrix
double matrixMinor(double** A, int rows, int columns, int i, int j)
{
	//If the matrix is square
	if (rows == columns)
	{
		//Minor is the determinant of the sub matrix
		double** sub_matrix = matrixSubmatrix(A, rows, columns, i, j);
		double minor = matrixDeterminant(sub_matrix, rows-1, columns-1);
		//Dispose the submatrix and return the minor
		matrixDispose(sub_matrix, rows-1);
		return minor;
	}
	else
	{
		printf("Minor of a non-square Matrix cannot be calculated");
		exit(EXIT_FAILURE);
	}
}

//The method for a cofactor of a matrix
double matrixCofactor(double** A, int rows, int columns, int i, int j)
{
	//If the matrix is square
	if (rows == columns)
	{
		//Cofactor is sign factor * minor
		int sign_factor = ((i + j) % 2 == 0) ? 1 : -1;
		return sign_factor * matrixMinor(A, rows, columns, i, j);
	}
	else
	{
		printf("Cofactor of a non-square Matrix cannot be calculated");
		exit(EXIT_FAILURE);
	}
}

//The method for the adjoint matrix of a matrix
double** matrixAdjoint(double** A, int rows, int columns)
{
	//If the matrix is square
	if (rows == columns)
	{
		//Get the cofactor matrix
		double** cofactor_matrix = initMatrix(rows, columns);
		//Fill the cofactor matrix with the cofactors
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < columns; j++)
			{
				cofactor_matrix[i][j] = matrixCofactor(A, rows, columns, i, j);
			}
		}
		//Get the adjoint matrix
		double** adj_matrix = matrixTranspose(cofactor_matrix, rows, columns);
		//Dispose the cofactor matrix
		matrixDispose(cofactor_matrix, rows);
		//Return the adjoint matrix
		return adj_matrix;
	}
	else
	{
		printf("Adjoint Matrix of a non-square Matrix cannot be calculated");
		exit(EXIT_FAILURE);
	}
}

//The method for the inverse of a matrix using its adjoint matrix
double** matrixInverseByAdjoint(double** A, int rows, int columns)
{
	//If the matrix is square
	if (rows == columns)
	{
		//Get the 1/determinant and the adjoint matrix
		double one_over_determinant = 1.0/matrixDeterminant(A, rows, columns);
		double** adj_matrix = matrixAdjoint(A, rows, columns);
		//Get the result matrix
		double** result_matrix = matrixScalarMultiplication(adj_matrix, rows, columns, one_over_determinant);
		//Dispose the adjoint matrix
		matrixDispose(adj_matrix, rows);
		//Return the inverse
		return result_matrix;
	}
	else
	{
		printf("Inverse of a non-square Matrix cannot be calculated");
		exit(EXIT_FAILURE);
	}
}

//The method for the inverse of a matrix
double** matrixInverse(double** A, int rows, int columns)
{
	//There is single method to find an inverse for now, new one will be added
	return matrixInverseByAdjoint(A, rows, columns);
}

//The method for the right inverse of a matrix
double** matrixRightInverse(double** A, int rows, int columns)
{
	//If the matrix is valid
	if (rows < columns)
	{
		double** Xt = matrixTranspose(A, rows, columns);
		double** XXt = matrixMultiplication(A, rows, columns, Xt, columns, rows);
		double** XXti = matrixInverse(XXt, rows, rows);
		//Result matrix : Xt x XXti
		double** result_matrix = matrixMultiplication(Xt, columns, rows, XXti, rows, rows);
		//Dispose the matrices
		matrixDispose(Xt, columns);
		matrixDispose(XXt, rows);
		matrixDispose(XXti, rows);
		//Return the result matrix
		return result_matrix;
	}
	else
	{
		printf("Right inverse of this Matrix cannot be calculated");
		exit(EXIT_FAILURE);
	}
}

//The method for the left inverse of a matrix
double** matrixLeftInverse(double** A, int rows, int columns)
{
	//If the matrix is valid
	if (rows > columns)
	{
		//Xt matrix, XtX matrix, and XtXi matrix
		double** Xt = matrixTranspose(A, rows, columns);
		double** XtX = matrixMultiplication(Xt, columns, rows, A, rows, columns);
		double** XtXi = matrixInverse(XtX, columns, columns);
		//Result matrix : XtXi x Xt
		double** result_matrix = matrixMultiplication(XtXi, columns, columns, Xt, columns, rows);
		//Dispose the matrices
		matrixDispose(Xt, columns);
		matrixDispose(XtX, columns);
		matrixDispose(XtXi, columns);
		//Return the result matrix
		return result_matrix;
	}
	else
	{
		printf("Left inverse of this Matrix cannot be calculated");
		exit(EXIT_FAILURE);
	}
}

//The method to dispose a Matrix
void matrixDispose(double** A, int rows)
{
	//Dispose the rows in the array
	for (int row_in_array_no = 0; row_in_array_no < rows; row_in_array_no++)
	{
		free(A[row_in_array_no]);
	}
	//Dispose the array
	free(A);
}

//The method to print the matrix
void printMatrix(double** A, int rows, int columns, int decimal_places)
{
	//Find the maximum length
	int max_length = 0;
	//Iterate on the Matrix to find the maximum length
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			//Find the current length
			int current_length = snprintf(NULL, 0, "%.*f", decimal_places, A[i][j]);
			//Set the new length
			if (current_length > max_length)
			{
				max_length = current_length;
			}
		}
	}
	//Iterate on the Matrix to print the Matrix
	for (int i = 0; i < rows; i++)
	{
		//Print the left border
		printf("| ");
		//Print the numbers on the row
		for (int j = 0; j < columns; j++)
		{
			//Find the current length
			int current_length = snprintf(NULL, 0, "%.*f", decimal_places, A[i][j]);
			//Print the spaces
			for (int print_space = 0; print_space < (max_length-current_length); print_space++)
			{
				printf(" ");
			}
			//Print the number itself
			printf("%.*f ", decimal_places, A[i][j]);
		}
		//Print the right border
		printf("|\n");
	}
}

/**
 * Methods to convert a vector to a matrix and vice versa
 */

//Method to convert a vector to a matrix
double** convertVectorMatrix(double* a, int n, int dispose)
{
	//Initialize the matrix to be returned
	double** matrix = initMatrix(n, 1);
	//Copy the elements on the vector
	for (int i = 0; i < n; i++)
	{
		matrix[i][0] = a[i];
	}
	//Dispose the vector
	if (dispose == 1)
	{
		free(a);
	}
	//Return the matrix
	return matrix;
}

//Method to convert a matrix to a vector
double* convertMatrixVector(double** A, int rows, int columns, int dispose)
{
	//Dimension check
	if (columns == 1)
	{
		//Initialize the vector to be returned
		double* vector = initVector(rows);
		//Copy the elements on the matrix
		for (int i = 0; i < rows; i++)
		{
			vector[i] = A[i][0];
		}
		//Dispose the matrix
		if (dispose == 1)
		{
			matrixDispose(A, rows);
		}
		//Return the vector
		return vector;
	}
	else
	{
		printf("Invalid dimensions for conversion");
		exit(EXIT_FAILURE);
	}
}

//Method to flatten a matrix into a vector
double* flatten(double** A, int rows, int columns, int dispose)
{
	//Initialize the flattened a
	double* a = initVector(rows * columns);
	//Copy the items
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			a[i * columns + j] = A[i][j];
		}
	}
	//Dispose the A if required
	if (dispose == 1)
	{
		matrixDispose(A, rows);
		A = NULL;
	}
	//Return the w
	return a;
}
