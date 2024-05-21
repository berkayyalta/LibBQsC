//Gradient descent class of LibBQsC by Berkay

#include "../../include/optimization/gradient_descent.h"

#include <stdio.h>
#include <stdlib.h>

#include "../../include/optimization/optimization_config.h"

//Constructor method of the gradient descent class
GradientDescent* initGradientDescent(double* w, int n)
{
	//Initialize the GradientDescent and handle any allocation failure
	GradientDescent* gradientDescent = malloc(sizeof(GradientDescent));
	if (gradientDescent == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Import the w and its size n
	gradientDescent->w = w;
	gradientDescent->n = n;
	//Assign the constant variables
	gradientDescent->learning_rate = gradient_descent_learning_rate;
	//Return the initialized ADAM optimizer
	return gradientDescent;
}

/**
 * Note : 	updateGradientDescent() method works element-wise instead of using the vectorized
 * 			operations maximise the efficiency.
 */

//Method to update the weights
double* updateGradientDescent(GradientDescent* gradientDescent, double* gradient, int n)
{
	//Check if the passed gradient is valid
	if (gradientDescent->n == n)
	{
		//Iterate over the w
		for (int index = 0; index < gradientDescent->n; index++)
		{
			//Gradient descent update rule
			gradientDescent->w[index] = gradientDescent->w[index] - gradientDescent->learning_rate * gradient[index];
		}
		//Return the updated w
		return gradientDescent->w;
	}
	//Throw exception otherwise
	else
	{
		printf("Invalid gradient for w");
		exit(EXIT_FAILURE);
	}
}

//Method to dispose a GradientDescent
void disposeGradientDescent(GradientDescent* gradientDescent, int dispose_w)
{
	//Dispose the w if required
	if (dispose_w == 1)
	{
		free(gradientDescent->w);
		gradientDescent->w = NULL;
	}
	//Dispose the GradientDescent itself
	free(gradientDescent);
	gradientDescent = NULL;
}

