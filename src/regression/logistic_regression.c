//Logistic regression class of LibBQsC by Berkay

#include "../../include/regression/logistic_regression.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../include/core/linear_algebra.h"
#include "../../include/metrics/regression_metrics.h"
#include "../../include/optimization/adam_optimizer.h"
#include "../../include/optimization/gradient_descent.h"
#include "../../include/statistics/statistics.h"

//Define the constant variables
int debugTrainingLogisticRegression = 0;

//Method to initialize a LogisticRegression struct
LogisticRegression* initLogisticRegression(double** X, double** Y, int samples, int features, int classes)
{
	//Initialize the logistic regression and handle any allocation failure
	LogisticRegression* regr = malloc(sizeof(LogisticRegression));
	if (regr == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Import the X and Y matrices
	regr->X = X;
	regr->Y = Y;
	//Import the dimensions of the matrices
	regr->samples = samples;
	regr->features = features;
	regr->classes = classes;
	//Initialize the W and dW matrices
	regr->W = initRandomMatrix(features, classes);
	regr->dW = initMatrix(features, classes);
	//Initialize the b and db matrices
	regr->b = initRandomVector(classes);
	regr->db = initVector(classes);
	//The P will be NULL initially
	regr->P = NULL;
	//Initialize the log loss as INT_MAX
	regr->log_loss = INT_MAX;
	//Return the initialized logistic regression
	return regr;
}

/**
 * Sigmoid and softmax functions :
 *
 * sigmoid(z) = 1 / (1 + exp(-z))
 *
 * softmax(zi) = exp(zi) / sum(exp(z))
 */

//Method to apply sigmoid to a vector
static double* sigmoid(double* z, int n)
{
	//Initialize the resulting vector p
	double* p = initVector(n);
	//Apply the sigmoid for all items of the z into the p
	for (int index = 0; index < n; index++)
	{
		p[index] = 1.0 / (1.0 + exp(-1.0 * z[index]));
	}
	//Return the p
	return p;
}

//Method to apply softmax to a vector
static double* softmax(double* z, int n)
{
	//Find the maximum number in the z to use for numerical stability
	double max_z = max(z, n);
	//max_z will be subtracted from every exp(zi)
	double sum_exp = 0.0;
	//Iterate to calculate the sum of exponentials
	for (int index = 0; index < n; index++)
	{
		sum_exp += exp(z[index] - max_z);
	}
	//Initialize the resulting vector p
	double* p = initVector(n);
	//Do exp(xi)/sum exp(x) to calculate the p
	for (int index = 0; index < n; index++)
	{
		p[index] = exp(z[index] - max_z) / sum_exp;
	}
	//Return the p
	return p;
}

/**
 * In calculating the P in this method, the ith row of the Z is disposed as ith
 * row of the P is generated out of the ith row of the Z.
 *
 * It would be impractical to store the P in the passed logistic regression struct like
 * the gradients are due to the dimensions of the P. A P matrix has dimensions of
 * (samples, classes) and the number of samples may be different when using this method to
 * make a prediction other than to train the model. Thus, this method returns a new P
 * matrix instead of updating an existing P matrix.
 */

//Method to generate the P : output of the logistic regression for the passed X matrix
static double** generateP(LogisticRegression* regr, double** X, int samples, int features)
{
	//If the passed X matrix is valid
	if (features == regr->features)
	{
		/**
		 * Calculate the P :
		 *
		 * Z = XW + b
		 * P = sigmoid/softmax(Z)
		 */
		//Initialize the Z : XW
		double** Z = initMatrix(samples, regr->classes);
		//Initialize the P and handle the allocation failure for the array (It will be double** but its rows won't be initialized yet)
		double** P = (double**) malloc (samples * sizeof(double*));
		if (P == NULL)
		{
			printf("Failed to allocate memory");
			exit(EXIT_FAILURE);
		}
		//Iterate over the rows of the Z
		for (int row_no = 0; row_no < samples; row_no++)
		{
			//Iterate over the columns of the Z
			for (int column_no = 0; column_no < regr->classes; column_no++)
			{
				double current_item = 0.0;
				//Iterate over the columns/rows of the X/W to perform the matrix multiplication
				for (int item_no = 0; item_no < features; item_no++)
				{
					current_item += X[row_no][item_no] * regr->W[item_no][column_no];
				}
				//Set the current item of the Z
				Z[row_no][column_no] = current_item + regr->b[column_no];
			}
			//After the current row is done, apply sigmoid into the current row of P if there is one class
			if (regr->classes == 1)
			{
				P[row_no] = sigmoid(Z[row_no], regr->classes);
			}
			//Or apply softmax otherwise
			else
			{
				P[row_no] = softmax(Z[row_no], regr->classes);
			}
			//The current row of the Z can be freed now since it won't be used again
			free(Z[row_no]);
			Z[row_no] = NULL;
		}
		//There are not any rows remaining in the Z
		free(Z);
		Z = NULL;
		//Return the P
		return P;
	}
	//Throw exception otherwise
	else
	{
		printf("Invalid X matrix.");
		exit(EXIT_FAILURE);
	}
}

/**
 * The methods to update the dW and db use the regr->P (the latest p) to update the
 * dW and db, so before calling them, update_P needs to be called first. Similarly,
 * the method update_W_b needs dW and db to be updated.
 */

//Method to update the P : XW + b
static void update_P(LogisticRegression* regr)
{
	//Generate the P
	double** P = generateP(regr, regr->X, regr->samples, regr->features);
	//Dispose any existing P
	if (regr->P != NULL)
	{
		matrixDispose(regr->P, regr->samples);
	}
	//Update the P
	regr->P = P;
}

//Method to update the dW : 1/m X^T (P - Y)
static void update_dW(LogisticRegression* regr)
{
	//Iterate over the rows of the dW
	for (int row_no = 0; row_no < regr->features; row_no++)
	{
		//Iterate over the columns of the dW
		for (int column_no = 0; column_no < regr->classes; column_no++)
		{
			double current_item = 0.0;
			//Iterate over the columns/rows of the X^T/(P-Y) to perform the matrix multiplication
			for (int item_no = 0; item_no < regr->samples; item_no++)
			{
				current_item += regr->X[item_no][row_no] * (regr->P[item_no][column_no] - regr->Y[item_no][column_no]);
			}
			//Set the current item of the gradient
			regr->dW[row_no][column_no] = (1.0/regr->samples) * current_item;
		}
	}
}

//Method to update the db : mean(P - Y)
static void update_db(LogisticRegression* regr)
{
	//Iterate over the items of the db
	for (int db_index = 0; db_index < regr->classes; db_index++)
	{
		double current_sum = 0.0;
		//Iterate to calculate the sum of the current column of the (P - Y)
		for (int item_no = 0; item_no < regr->samples; item_no++)
		{
			current_sum += (regr->P[item_no][db_index] - regr->Y[item_no][db_index]);
		}
		//Set the current item of the db
		regr->db[db_index] = (1.0/regr->samples) * current_sum;
	}
}

//Method to train a logistic regression
void trainLogisticRegression(LogisticRegression* regr, Optimizer optimizer, int max_iterations, double threshold)
{
	//Flatten the W into a new vector that is to be of the optimizer
	double* w = flatten(regr->W, regr->features, regr->classes, 0);
	/**
	 * Initialize the optimizers. Initialize the one for the W out of the flattened W
	 * w, and initialize the one for b passing the b of the logistic regression itself.
	 */
	//Declare the optimizers
	void* optimizer_w;
	void* optimizer_b;
	//Instantiate the optimizers if gradient descent will be used
	if (optimizer == GRADIENT_DESCENT)
	{
		optimizer_w = initGradientDescent(w, regr->features * regr->classes);
		optimizer_b = initGradientDescent(regr->b, regr->classes);
	}
	//Initialize the optimizers if ADAM optimizer will be used
	else if (optimizer == ADAM_OPTIMIZER)
	{
		optimizer_w = initADAM(w, regr->features * regr->classes);
		optimizer_b = initADAM(regr->b, regr->classes);
	}
	/**
	 * Begin the iteration : iterates max_iteration times unless the converge
	 * checking if statement breaks the loop
	 */
	for (int t = 0; t < max_iterations; t++)
	{
		/**
		 * Update the P and the gradients calling these methods so the
		 * weights and the biasas can be updated using the gradients
		 */
		update_P(regr);
		update_dW(regr);
		update_db(regr);
		/**
		 * Calculate the current log loss and check the converge
		 * Print the current t and loss if the debug mode is enabled
		 */
		double loss_current = logLossMatrix(regr->Y, regr->P, regr->samples, regr->classes);
		//Check converge and update the log loss of the logistic regression struct after that
		if (fabs(regr->log_loss - loss_current) < threshold)
		{
			break;
		}
		regr->log_loss = loss_current;
		//Print the current t and loss if the debugTraining is 1
		if (debugTrainingLogisticRegression == 1)
		{
			printf("t : %d , loss : %f\n", t, loss_current);
		}
		/**
		 * Update the weights and the biases. Calling the update method of the optimizer
		 * is sufficient for the biases since it is already a vector, but the returned w
		 * needs to be copied into the W of the logistic regression. Also, the gradient of
		 * the W needs to be flattened to be able to be passed to the update methods.
		 */
		//Flatten the dW to update the W and declare the updated_w
		double* dw = flatten(regr->dW, regr->features, regr->classes, 0);
		double* updated_w;
		//Call the update methods for gradient descent
		if (optimizer == GRADIENT_DESCENT)
		{
			updated_w = updateGradientDescent(optimizer_w, dw, regr->features * regr->classes);
			updateGradientDescent(optimizer_b, regr->db, regr->classes);
		}
		//Call the update methods for ADAM optimizer
		else if (optimizer == ADAM_OPTIMIZER)
		{
			updated_w = updateADAM(optimizer_w, dw, regr->features * regr->classes);
			updateADAM(optimizer_b, regr->db, regr->classes);
		}
		//Copy the updated w into the W of the logistic regression to update it
		int updated_w_index = 0;
		for (int row_no = 0; row_no < regr->features; row_no++)
		{
			for (int column_no = 0; column_no < regr->classes; column_no++)
			{
				regr->W[row_no][column_no] = updated_w[updated_w_index];
				updated_w_index += 1;
			}
		}
		//Dispose the dw
		free(dw);
		dw = NULL;
	}
	/**
	 * The second parameter of the disposeADAM() is whether the w of the ADAM will be disposed.
	 * The w of the first ADAM should be disposed since it is a new vector that is the flattened of
	 * the original W, but the w of the second ADAM shouldn't be disposed since it is the b of the
	 * logistic regression itself.
	 */
	//Dispose the optimizers after the optimization if gradient descent will be used
	if (optimizer == GRADIENT_DESCENT)
	{
		disposeGradientDescent(optimizer_w, 1);
		disposeGradientDescent(optimizer_b, 0);
	}
	//Dispose the optimizers after the optimization if ADAM optimizer will be used
	else if (optimizer == ADAM_OPTIMIZER)
	{
		disposeADAM(optimizer_w, 1);
		disposeADAM(optimizer_b, 0);
	}
}

//Method to make a prediction
double** predictLogisticRegression(LogisticRegression* regr, double** X, int samples, int features)
{
	//Get the P calculated using the passed X
	double** P = generateP(regr, X, samples, features);
	//Return the P
	return P;
}

//Method to print a logistic regression
void printLogisticRegression(LogisticRegression* regr, int decimal_places)
{
	//Print the title
	if (regr->classes == 1)
	{
		printf("Logistic Regression Model : \n");
	}
	else
	{
		printf("Multinomial Logistic Regression Model : \n");
	}
	//Print the samples, features, and classes
	printf("- Number of Samples : %d\n", regr->samples);
	printf("- Number of Features : %d\n", regr->features);
	printf("- Number of Classes : %d\n", regr->classes);
	//Print the weight matrix W
	printf("- Weight Matrix (W) : \n");
	printMatrix(regr->W, regr->features, regr->classes, decimal_places);
	//Print the bias vector b
	printf("- Bias Vector (b) : \n");
	printVector(regr->b, regr->classes, decimal_places);
	//Print the final loss
	printf("- Final Loss : %.*f\n", decimal_places, regr->log_loss);
}

//Method to dispose a LogisticRegression struct
void disposeLogisticRegression(LogisticRegression* regr)
{
	//Dispose the W and the dW of the logistic regression
	matrixDispose(regr->W, regr->features);
	regr->W = NULL;
	matrixDispose(regr->dW, regr->features);
	regr->dW = NULL;
	//Dispose the b and the db of the logistic regression
	free(regr->b);
	regr->b = NULL;
	free(regr->db);
	regr->db = NULL;
	//Dispose the P of the logistic regression
	matrixDispose(regr->P, regr->samples);
	regr->P = NULL;
	//Dispose the logistic regression itself
	free(regr);
	regr = NULL;
}

