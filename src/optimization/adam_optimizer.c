//ADAM optimizer class of LibBQsC by Berkay

#include "../../include/optimization/adam_optimizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../include/core/linear_algebra.h"
#include "../../include/optimization/optimization_config.h"

//Constructor method of the ADAM optimizer class
ADAM* initADAM(double* w, int n)
{
	//Initialize the ADAM optimizer and handle any allocation failure
	ADAM* adam = malloc(sizeof(ADAM));
	if (adam == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Import the w and its size n
	adam->w = w;
	adam->n = n;
	//Assign the constant variables
	adam->learning_rate = adam_learning_rate;
	adam->beta_1 = adam_beta_1;
	adam->beta_2 = adam_beta_2;
	adam->epsilon = adam_epsilon;
	//Initial step (t=0)
	adam->m = initZeroVector(n);
	adam->v = initZeroVector(n);
	adam->t = 1;
	//Return the initialized ADAM optimizer
	return adam;
}

/**
 * Note : 	updateADAM() method works element-wise instead of using the vectorized
 * 			operations maximise the efficiency.
 */

//Method to update the weights
double* updateADAM(ADAM* adam, double* gradient, int n)
{
	//Check if the passed gradient is valid
	if (adam->n == n)
	{
		//Iterate over the w, v and m which are all same size
		for (int index = 0; index < adam->n; index++)
		{
			//Update the moment estimates
			adam->m[index] = adam->beta_1 * adam->m[index] + (1.0 - adam->beta_1) * gradient[index];
			adam->v[index] = adam->beta_2 * adam->v[index] + (1.0 - adam->beta_2) * pow(gradient[index], 2);
			//Calculate the bias-corrected moment estimates
			double m_hat = adam->m[index] / (1.0 - pow(adam->beta_1, adam->t));
			double v_hat = adam->v[index] / (1.0 - pow(adam->beta_2, adam->t));
			//ADAM update rule
			adam->w[index] = adam->w[index] - adam->learning_rate * m_hat / (pow(v_hat, 0.5) + adam->epsilon);
		}
		//Increase the t
		adam->t += 1;
		//Return the updated w
		return adam->w;
	}
	//Throw exception otherwise
	else
	{
		printf("Invalid gradient for w");
		exit(EXIT_FAILURE);
	}
}

//Method to dispose an ADAM optimizer struct
void disposeADAM(ADAM* adam, int dispose_w)
{
	//Dispose moment estimates of the ADAM optimizer
	free(adam->m);
	adam->m = NULL;
	free(adam->v);
	adam->v = NULL;
	//Dispose the w as well if required
	if (dispose_w == 1)
	{
		free(adam->w);
		adam->w = NULL;
	}
	//Dispose the ADAM optimizer itself
	free(adam);
	adam = NULL;
}

