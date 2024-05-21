//Regression metrics class of LibBQsC by Berkay

#include "../../include/metrics/regression_metrics.h"

#include <math.h>

//Method to calculate a mean squared error
double meanSquaredError(double *y_true, double *y_pred, int n)
{
	//Initialize the sum
	double sum = 0;
	//Calculate the sum
	for (int i = 0; i < n; i++)
	{
		//The square of y_pred - y_true
		sum += pow((y_pred[i] - y_true[i]), 2);
	}
	//Return the MSE
	return sum / n;
}

//Method to calculate a mean absolute error
double meanAbsoluteError(double *y_true, double *y_pred, int n)
{
	//Initialize the sum
	double sum = 0;
	//Calculate the sum
	for (int i = 0; i < n; i++)
	{
		//The square of y_pred - y_true
		sum += fabs(y_pred[i] - y_true[i]);
	}
	//Return the MAE
	return sum / n;
}

//Method to calculate a log loss
double logLoss(double *y_true, double *y_pred, int n)
{
	//Initialize the sum
	double sum = 0;
	//Calculate the sum
	for (int i = 0; i < n; i++)
	{
		sum += y_true[i] * log(y_pred[i]) + (1.0 - y_true[i]) * log(1.0 - y_pred[i]);
	}
	//Return the log loss
	return -1.0 * sum / n;
}

//Method to calculate a log loss for matrices
double logLossMatrix(double** y_true, double** y_pred, int samples, int classes)
{
	//Small value to avoid numerical instability (log(0))
	double epsilon = 1e-15;
	//Initialize the sum
	double sum = 0.0;
	//Calculate the sum
	for (int i = 0; i < samples; i++)
	{
		for (int j = 0; j < classes; j++)
		{
			//Clip the actual p
			double current_p = fmax(epsilon, fmin(1.0 - epsilon, y_pred[i][j]));
			sum -= y_true[i][j] * log(current_p);
		}
	}
	//Return the log loss
	return sum / samples;
}

