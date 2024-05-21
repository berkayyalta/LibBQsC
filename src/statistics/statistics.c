//Statistics class of LibBQsC by Berkay

#include "../../include/statistics/statistics.h"

#include <math.h>

//Method to calculate the sum of an array
double sum(double* array, int n)
{
	double sum = 0.0;
	//Calculate the sum
	for (int i = 0; i < n; i++)
	{
		sum += array[i];
	}
	//Return the sum
	return sum;
}

//Method to calculate the mean of an array
double mean(double* array, int n)
{
	//Calculate and return the mean
	return sum(array, n)/n;
}

//Method to get the median of an array
double median(double* array, int n)
{
	double median = 0.0;
	//Even sized array
	if (n%2 == 0)
	{
		median = (array[n/2] + array[(n/2)-1]) / 2.0;
	}
	//Odd sized array
	else
	{
		median = array[n/2];
	}
	//Return the median
	return median;
}

//Method to get the minimum value in an array
double min(double* array, int n)
{
	double min = array[0];
	//Find the minimum
	for (int i = 1; i < n; i++)
	{
		if (array[i] < min)
		{
			min = array[i];
		}
	}
	//Return the minimum
	return min;
}

//Method to get the maximum value in an array
double max(double* array, int n)
{
	double max = array[0];
	//Find the maximum
	for (int i = 1; i < n; i++)
	{
		if (array[i] > max)
		{
			max = array[i];
		}
	}
	//Return the maximum
	return max;
}

//Method to calculate the range of an array
double range(double* array, int n)
{
	double min = array[0];
	double max = array[0];
	//Find the minimum and the maximum
	for (int i = 1; i < n; i++)
	{
		if (array[i] < min)
		{
			min = array[i];
		}
		else if (array[i] > max)
		{
			max = array[i];
		}
	}
	//Return the range
	return max - min;
}

//Method to calculate the standard deviation of an array
double standardDeviation(double* array, int n)
{
	double mean_array = mean(array, n);
	//Calculate the sum of squared deviations
	double sum_deviation_square = 0.0;
	for (int i = 0; i < n; i++)
	{
		sum_deviation_square += pow(mean_array - array[i], 2);
	}
	//Divide it by (n-1) : sample
	sum_deviation_square /= (n-1);
	//Return the square root of it
	return pow(sum_deviation_square, 0.5);
}
