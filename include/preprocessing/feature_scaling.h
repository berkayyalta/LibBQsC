//Feature scaling class of LibBQsC by Berkay

#ifndef FEATURE_SCALING_H
#define FEATURE_SCALING_H

/**
 * Method for min-max scaling
 *
 * x_scaled = (x - x_min) / (x_max - x_min)
 *
 * @param	X_input 	X input array : matrix
 * @param	min			vector of minimums of columns of X input matrix
 * @param	range		vector of ranges of columns of X input matrix
 * @param	samples		number of samples : rows
 * @param	features	number of features : columns
 * @return				scaled X
 */
double** minMaxScale(double** X_input, double* min, double* range, int samples, int features);

/**
 * Method to revert min-max scaling
 *
 * x = ((x_max - x_min) * x_scaled) + x_min
 *
 * @param	X_input 	X scaled array : matrix
 * @param	min			vector of minimums of columns of X input matrix
 * @param	range		vector of ranges of columns of X input matrix
 * @param	samples		number of samples : rows
 * @param	features	number of features : columns
 * @return				X
 */
double** inverseMinMaxScale(double** X_scaled, double* min, double* range, int samples, int features);

/**
 * Method for standardization
 *
 * x_scaled = (x - x_mean) / (x_standard_deviation)
 *
 * @param	X_input 			X input array : matrix
 * @param	mean				vector of means of columns of X input matrix
 * @param	standard_deviation	vector of standard deviations of columns of X input matrix
 * @param	samples				number of samples : rows
 * @param	features			number of features : columns
 * @return						scaled X
 */
double** standardize(double** X_input, double* mean, double* standard_deviation, int samples, int features);

/**
 * Method to revert standardization
 *
 * x = (x_standard_deviation * x_scaled) + x_mean
 *
 * @param	X_input 			X scaled array : matrix
 * @param	mean				vector of means of columns of X input matrix
 * @param	standard_deviation	vector of standard deviations of columns of X input matrix
 * @param	samples				number of samples : rows
 * @param	features			number of features : columns
 * @return						X
 */
double** inverseStandardize(double** X_scaled, double* mean, double* standard_deviation, int samples, int features);

#endif //FEATURE_SCALING_H
