//Logistic regression class of LibBQsC by Berkay

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "../optimization/optimization_config.h"

//Extern the constant variables
extern int debugTrainingLogisticRegression;

/**
 * The logistic regression struct
 */
typedef struct
{
	//X and Y matrices
	double** X;
	double** Y;
	//Dimensions of the matrices
	int samples;
	int features;
	int classes;
	//Weights and their gradients
	double** W;
	double** dW;
	//Bias terms and their gradients
	double* b;
	double* db;
	//Last prediction made to be used to calculate the loss in the training
	double** P;
	//Log loss of the model
	double log_loss;
}
LogisticRegression;

/**
 * Method to initialize a LogisticRegression struct
 *
 * @param	X			X feature matrix
 * @param 	Y			Y matrix
 * @param	samples		number of samples in the X and Y matrices
 * @param	features	number of features in the X matrix
 * @param	classes		number of classes in the Y matrix
 * @return				pointer to the initialized LogisticRegression
 */
LogisticRegression* initLogisticRegression(double** X, double** Y, int samples, int features, int classes);

/**
 * Method to train a logistic regression
 *
 * @param 	regr			logistic regression to be trained
 * @param	max_iterations	maximum number of iterations
 * @param	threshold		training will stop if the change in the loss
 * 							function is smaller than the threshold
 */
void trainLogisticRegression(LogisticRegression* regr, Optimizer optimizer, int max_iterations, double threshold);

/**
 * Method to make a prediction
 *
 * @param regr		trained LogisticRegression
 * @param X			data points to be predicted
 * @param samples	number of data points in the X to be predicted
 * @param features	number of features in the X to be predicted
 * @return			output of the LogisticRegression
 */
double** predictLogisticRegression(LogisticRegression* regr, double** X, int samples, int features);

/**
 * Method to print a logistic regression
 *
 * @param 	regr			the logistic regression to be printed
 * @param 	decimal_places	number of the decimal places to print
 */
void printLogisticRegression(LogisticRegression* regr, int decimal_places);

/**
 * Method to dispose a LogisticRegression struct
 *
 * @param	regr	LogisticRegression to be disposed
 */
void disposeLogisticRegression(LogisticRegression* regr);

#endif //LOGISTIC_REGRESSION_H
