//Feature scaling class of LibBQsC by Berkay

#include "../../include/preprocessing/feature_scaling.h"

#include <math.h>

#include "../../include/core/linear_algebra.h"

//Method for min-max scaling
double** minMaxScale(double** X_input, double* min, double* range, int samples, int features)
{
	//Initialize the empty scaled X matrix
	double** X_scaled = initMatrix(samples, features);
	//Iterate over the matrix
	for (int i = 0; i < samples; i++)
	{
		//Do (x - min_x)/(range_x)
		for (int j = 0; j < features; j++)
		{
			//Min and range are vectors of the minimums and ranges of the columns (min[n] is the minimum of the nth column)
			X_scaled[i][j] -= min[j];
			X_scaled[i][j] /= range[j];
		}
	}
	//Return the scaled X
	return X_scaled;
}

//Method to revert min-max scaling
double** inverseMinMaxScale(double** X_scaled, double* min, double* range, int samples, int features)
{
	//Initialize the empty X matrix
	double** X = initMatrix(samples, features);
	//Iterate over the matrix
	for (int i = 0; i < samples; i++)
	{
		//Do ((x_max - x_min) * x_scaled) + x_min
		for (int j = 0; j < features; j++)
		{
			//Min and range are vectors of the minimums and ranges of the columns (min[n] is the minimum of the nth column)
			X[i][j] *= range[j];
			X[i][j] += min[j];
		}
	}
	//Return the scaled X
	return X;
}

//Method for standardization
double** standardize(double** X_input, double* mean, double* standard_deviation, int samples, int features)
{
	//Initialize the empty scaled X matrix
	double** X_scaled = initMatrix(samples, features);
	//Iterate over the matrix
	for (int i = 0; i < samples; i++)
	{
		//Do (x - mean_x)/(standard_deviation_x)
		for (int j = 0; j < features; j++)
		{
			//Mean and standard_deviation are vectors of the means and standard deviations of the columns (mean[n] is the mean of the nth column)
			X_scaled[i][j] -= mean[j];
			X_scaled[i][j] /= standard_deviation[j];

			//Temp
			if (isinf(X_scaled[i][j]))
			{
				X_scaled[i][j] = 0.0;
			}

		}
	}
	//Return the scaled X
	return X_scaled;
}

//Method to revert standardization
double** inverseStandardize(double** X_scaled, double* mean, double* standard_deviation, int samples, int features)
{
	//Initialize the empty X matrix
	double** X = initMatrix(samples, features);
	//Iterate over the matrix
	for (int i = 0; i < samples; i++)
	{
		//Do (x_standard_deviation * x_scaled) + x_mean
		for (int j = 0; j < features; j++)
		{
			//Mean and standard_deviation are vectors of the means and standard deviations of the columns (mean[n] is the mean of the nth column)
			X[i][j] *= standard_deviation[j];
			X[i][j] += mean[j];
		}
	}
	//Return the scaled X
	return X;
}
