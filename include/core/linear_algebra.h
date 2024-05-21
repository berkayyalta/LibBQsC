//Linear algebra class of LibBQsC by Berkay

#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

/**
 * Note :	Methods in this class do not handle possible exceptions,
 * 			such as division by zero, except some dimension exceptions.
 *
 * Note : 	Methods except the conversion methods in this class do not
 * 			dispose the passed arrays. So, the passed arrays should be
 * 			disposed by the user if are no longer needed.
 *
 * Note : 	Some other methods such as LU decomposition are likely
 * 			to be added to this class later. When these methods are
 * 			added, the methods like matrixDeterminant() and
 * 			matrixInverse() will handle the selection of the method
 * 			to be used.
 */

/**
 * Vector methods of the linear algebra class
 */

/*
 * Method to initialize a vector of zeros
 *
 * @param	n	size of the vector
 * @return 		the vector
 */
double* initVector(int n);

/*
 * Method to initialize a vector of zeros - separate method for clarity
 *
 * @param	n	size of the vector
 * @return 		the vector
 */
double* initZeroVector(int n);

/*
 * Method to initialize a vector of random numbers
 *
 * @param	n	size of the vector
 * @return 		the vector
 */
double* initRandomVector(int n);

/**
 * Method to append a number to a vector
 *
 * @param 	vector 	vector to which the number will be appended
 * @param 	n 		size of the vector
 * @param 	number  number to be appended
 * @param 	index 	index to append the number
 * @param	dispose	existing vector will be disposed if 1
 * @return 			the resulting vector
 */
double* vectorAppend(double* vector, int n, double number, int index, int dispose);

/**
 * Method for vector addition
 *
 * @param 	a 	first vector
 * @param 	b 	second vector
 * @param 	n 	size of the vectors
 * @return 		the result
 */
double* vectorAddition(double* a, double* b, int n);

/**
 * Method for vector subtraction
 *
 * @param 	a 	first vector
 * @param 	b 	second vector
 * @param 	n 	size of the vectors
 * @return 		the result
 */
double* vectorSubtraction(double* a, double* b, int n);

/**
 * Method for scalar multiplication
 *
 * @param	vector	vector
 * @param	n		size of the vector
 * @param	scalar	scalar
 * @return 			the result
 */
double* vectorScalarMultiplication(double* vector, int n, double scalar);

/**
 * Method for vector dot product
 *
 * @param 	a 	first vector
 * @param 	b 	second vector
 * @param 	n 	size of the vectors
 * @return 		the result
 */
double vectorDotProduct(double* a, double* b, int n);

/**
 * Method to print a vector
 *
 * @param	a				vector to be printed
 * @param	n				size of the vector
 * @param	decimal_places	number of decimal places to print
 */
void printVector(double* a, int n, int decimal_places);

/**
 * Matrix methods of the linear algebra class
 */

/**
 * Method for generating double**
 *
 * @param	rows	number of rows to be in the matrix
 * @param 	columns	number of columns to be in the matrix
 * @return			initialized matrix - double**
 */
double** initMatrix(int rows, int columns);

/**
 * Method for generating double** with zeroes
 *
 * @param	rows	number of rows to be in the matrix
 * @param 	columns	number of columns to be in the matrix
 * @return			initialized matrix - double**
 */
double** initZeroMatrix(int rows, int columns);

/**
 * Method for generating double** with random number between 0 and 1
 *
 * @param	rows	number of rows to be in the matrix
 * @param 	columns	number of columns to be in the matrix
 * @return			initialized matrix - double**
 */
double** initRandomMatrix(int rows, int columns);

/**
 * Method to get a row from a matrix; returns the clone of the row
 *
 * @param	A		matrix whose row will be returned
 * @param	rows	number of rows in the matrix
 * @param	columns	number of columns in the matrix
 * @param	index	index of the row to be returned
 * @return 			the row : vector
 */
double* matrixGetRow(double** A, int rows, int columns, int index);

/**
 * Method to get a column from a matrix; returns the clone of the column
 *
 * @param	A		matrix whose column will be returned
 * @param	rows	number of rows in the matrix
 * @param	columns	number of columns in the matrix
 * @param	index	index of the column to be returned
 * @return 			the column : vector
 */
double* matrixGetColumn(double** A, int rows, int columns, int index);

/**
 * Method for matrix addition
 *
 * @param	A		first matrix
 * @param	B 		second matrix
 * @param	rows	number of rows in the matrices
 * @param	columns number of columns in the matrices
 * @return			resulting matrix
 */
double** matrixAddition(double** A, double** B, int rows, int columns);

/**
 * Method for matrix subtraction
 *
 * @param	A		first matrix
 * @param	B 		second matrix
 * @param	rows	number of rows in the matrices
 * @param	columns number of columns in the matrices
 * @return			resulting matrix
 */
double** matrixSubtraction(double** A, double** B, int rows, int columns);

/**
 * Method for matrix scalar multiplication
 *
 * @param	A		matrix
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @param 	scalar	scalar to multiply the matrix by
 * @return			resulting matrix
 */
double** matrixScalarMultiplication(double** A, int rows, int columns, double scalar);

/**
 * Method for matrix multiplication
 *
 * @param	A			first matrix
 * @param 	rows_A		number of rows in the A matrix
 * @param	columns_A 	number of columns in the A matrix
 * @param	B 			second matrix
 * @param	rows_B		number of rows in the B matrix
 * @param	columns_B	number of columns in the B matrix
 * @return				resulting matrix
 */
double** matrixMultiplication(double** A, int rows_A, int columns_A, double** B, int rows_B, int columns_B);

/**
 * Method for matrix transpose
 *
 * @param	A		the matrix
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @return			resulting matrix
 */
double** matrixTranspose(double** A, int rows, int columns);

/**
 * Method for sub matrix
 *
 * @param	A		the matrix
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @param	i		index of the row to exclude
 * @param	j		index of the column to exclude
 * @return			resulting matrix
 */
double** matrixSubmatrix(double** A, int rows, int columns, int i, int j);

/**
 * Method for calculating a determinant using Laplace Expansion
 *
 * @param	A		the matrix whose determinant will be calculated
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @return 			determinant of the matrix
 */
double matrixDeterminantLaplaceExpansion(double** A, int rows, int columns);

/**
 * Method for calculating a determinant
 *
 * @param	A		the matrix whose determinant will be calculated
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @return 			determinant of the matrix
 */
double matrixDeterminant(double** A, int rows, int columns);

/**
 * Method for calculating a minor of a matrix
 *
 * @param	A		the matrix
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @param	i		index of the row to exclude
 * @param	j		index of the column to exclude
 * @return			minor of the matrix on the specified index
 */
double matrixMinor(double** A, int rows, int columns, int i, int j);

/**
 * Method for calculating a cofactor of a matrix
 *
 * @param	A		the matrix
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @param	i		index of the row to exclude
 * @param	j		index of the column to exclude
 * @return			cofactor of the matrix on the specified index
 */
double matrixCofactor(double** A, int rows, int columns, int i, int j);

/**
 * Method for calculating an adjoint matrix
 *
 * @param	A		the matrix whose adjoint will be calculated
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @return 			adjoint of the matrix
 */
double** matrixAdjoint(double** A, int rows, int columns);

/**
 * Method for calculating the inverse of a matrix using its adjoint matrix
 *
 * @param	A		the matrix whose inverse will be calculated
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @return 			inverse of the matrix
 */
double** matrixInverseByAdjoint(double** A, int rows, int columns);

/**
 * Method for calculating the inverse of a matrix
 *
 * @param	A		the matrix whose inverse will be calculated
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @return 			inverse of the matrix
 */
double** matrixInverse(double** A, int rows, int columns);

/**
 * Method for calculating the right inverse of a matrix
 *
 * @param	A		the matrix whose right inverse will be calculated
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @return 			right inverse of the matrix
 */
double** matrixRightInverse(double** A, int rows, int columns);

/**
 * Method for calculating the left inverse of a matrix
 *
 * @param	A		the matrix whose left inverse will be calculated
 * @param	rows	number of rows in the matrix
 * @param	columns number of columns in the matrix
 * @return 			left inverse of the matrix
 */
double** matrixLeftInverse(double** A, int rows, int columns);

/**
 * Method to dispose a Matrix
 *
 * @param	A		matrix to be disposed
 * @param	rows	number of rows in the matrix
 */
void matrixDispose(double** A, int rows);

/**
 * Method to print a matrix
 *
 * @param	A				matrix to print
 * @param	rows			number of rows in the matrix
 * @param	columns			number of columns in the matrix
 * @param	decimal_places	number of decimal places to print
 */
void printMatrix(double** A, int rows, int columns, int decimal_places);

/**
 * Methods to convert a vector to a matrix and vice versa
 *
 * These methods dispose the passed arrays
 */

/**
 * Method to convert a vector to a matrix
 *
 * @param	a		vector to be converted
 * @param	n		size of the vector
 * @param 	dispose	1 if the matrix will be disposed
 * @return			matrix of dimensions (n, 1)
 */
double** convertVectorMatrix(double* a, int n, int dispose);

/**
 * Method to convert a matrix to a vector
 *
 * Will convert if the matrix has already one column
 *
 * @param	A		matrix to be converted
 * @param	rows	number of rows in the matrix
 * @param	columns	number of columns in the matrix
 * @param 	dispose	1 if the matrix will be disposed
 * @return			vector of size rows
 */
double* convertMatrixVector(double** A, int rows, int columns, int dispose);

/**
 * Method to flatten a matrix into a vector
 *
 * @param 	A			matrix to be flattened
 * @param 	rows		number of rows in the matrix
 * @param	 columns	number of columns in the matrix
 * @param 	dispose		1 if the matrix will be disposed
 * @return
 */
double* flatten(double** A, int rows, int columns, int dispose);

#endif //LINEAR_ALGEBRA_H
