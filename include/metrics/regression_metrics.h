//Regression metrics class of LibBQsC by Berkay

#ifndef REGRESSION_METRICS_H
#define REGRESSION_METRICS_H

/**
 * Note : 	Some other features such as weights for MSE are likely
 * 			to be added.
 *
 * Note : 	Methods in this class assume that the passed parameters
 * 			are valid so possible errors such as division by 0 are
 * 			not handled
 *
 * Note : 	This class may not be fully optimized yet.
 */

/**
 * The enum of loss functions
 */
typedef enum
{
	MSE,
	MAE,
	LOGLOSS
}
regressionLossFunction;

/**
 * Method to calculate a mean squared error
 *
 * @param	y_true	true y values
 * @param 	y_pred 	predicted y values
 * @param 	n 	 	number of samples
 * @return 	MSE
 */
double meanSquaredError(double* y_true, double*y_pred, int n);

/**
 * Method to calculate a mean absolute error
 *
 * @param	y_true  true y values
 * @param 	y_pred  predicted y values
 * @param 	n 	 	number of samples
 * @return 	MAE
 */
double meanAbsoluteError(double* y_true, double* y_pred, int n);

/**
 * Method to calculate a log loss
 *
 * @param	y_true  true y values
 * @param 	y_pred  predicted y values
 * @param 	n 	 	number of samples
 * @return 	log loss
 */
double logLoss(double* y_true, double* y_pred, int n);

/**
 * Method to calculate a log loss for matrices
 *
 * @param	y_true 		true y values
 * @param 	y_pred 		predicted y values
 * @param 	rows 		number of rows in the matricex
 * @param	columns		number of columns in the matricex
 * @return	   			log loss
 */
double logLossMatrix(double** y_true, double** y_pred, int samples, int classes);

#endif //REGRESSION_METRICS_H
