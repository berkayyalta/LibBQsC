//Statistics class of LibBQsC by Berkay

#ifndef STATISTICS_H
#define STATISTICS_H

/**
 * Method to calculate the sum of an array
 *
 * @param	array	array whose sum will be calculated
 * @param	n		length of the array
 * @return			sum of the array
 */
double sum(double* array, int n);

/**
 * Method to calculate the mean of an array
 *
 * @param	array	array whose mean will be calculated
 * @param	n		length of the array
 * @return			mean of the array
 */
double mean(double* array, int n);

/**
 * Method to get the median of an array
 *
 * @param	array	array whose median will be got
 * @param	n		length of the array
 * @return			median of the array
 */
double median(double* array, int n);

/**
 * Method to get the minimum value in an array
 *
 * @param	array	array whose minimum will be got
 * @param	n		length of the array
 * @return			minimum value in the array
 */
double min(double* array, int n);

/**
 * Method to get the maximum value in an array
 *
 * @param	array	array whose maximum will be got
 * @param	n		length of the array
 * @return			maximum value in the array
 */
double max(double* array, int n);

/**
 * Method to calculate the range of an array
 *
 * @param	array	array whose range will be calculated
 * @param	n		length of the array
 * @return			range of the array
 */
double range(double* array, int n);

/**
 * Method to calculate the standard deviation of an array
 *
 * @param	array	array whose standard deviation will be calculated
 * @param	n		length of the array
 * @return			standard deviation of the array
 */
double standardDeviation(double* array, int n);

#endif //STATISTICS_H
