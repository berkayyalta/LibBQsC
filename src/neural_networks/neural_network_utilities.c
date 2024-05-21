//Neural network utilities class of LibBQsC by Berkay

#include "../../include/neural_networks/neural_network_utilities.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../include/core/linear_algebra.h"

//Method to apply the activation function
double activationFunction(double z, Activation activation)
{
	//Initialize the result
	double a;
	//If RELU will be used
	if (activation == RELU)
	{
		a = (z > 0) ? z : 0;
	}
	//If sigmoid will be used
	else if (activation == SIGMOID)
	{
		a = 1 / (1 + exp(-z));
	}
	//If tanh will be used
	else if (activation == TANH)
	{
		a = tanh(z);
	}
	//Return the result
	return a;
}

//Method to apply the derivative of the activation function
double activationFunctionDerivative(double z, Activation activation)
{
	//Initialize the result
	double dS;
	//If RELU will be used
	if (activation == RELU)
	{
		dS = (z > 0) ? 1 : 0;
	}
	//If sigmoid will be used
	else if (activation == SIGMOID)
	{
		double sigmoid_z = activationFunction(z, activation);
		dS = sigmoid_z * (1 - sigmoid_z);
	}
	//If tanh will be used
	else if (activation == TANH)
	{
		double tanh_z = activationFunction(z, activation);
		dS = 1 - tanh_z * tanh_z;
	}
	//Return the result
	return dS;
}

/*
 * initB method initialized the proper B matrix for an ANNLayer. B matrix should be
 * consist of same random rows.
 */

//Static method to initialize the B matrix of an ANNLayer
static double** initB(int samples, int neurons)
{
	//Initialize the double** B matrix and handle any allocation failure
	double** B = (double**) malloc (samples * sizeof(double*));
	if (B == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Seed the random number generator
	srand(time(NULL));
	//Define the rows of the B matrix
	for (int i = 0; i < samples; i++)
	{
		B[i] = (double*) malloc (neurons * sizeof(double));
		//Handle the allocation failure for the current row of the array
		if (B[i] == NULL)
		{
			printf("Failed to allocate memory");
			exit(EXIT_FAILURE);
		}
		//Define the items of the current row
		for (int j = 0; j < neurons; j++)
		{
			//If the current row is the first row
			if (i == 0)
			{
				//Define the random number
				B[i][j] = (double)rand() / RAND_MAX;
			}
			//Otherwise
			else
			{
				//Copy the numbers from the first row
				B[i][j] = B[0][j];
			}
		}
	}
	//Return the B matrix
	return B;
}

//Method to initialze an ANNLayer
ANNLayer* initANNLayer(int samples, int neurons_previous, int neurons, LayerType layer_type, Activation activation)
{
	//Initialize the ANN and handle any allocation failure
	ANNLayer* ann_layer = malloc(sizeof(ANNLayer));
	if (ann_layer == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Import the dimensions
	ann_layer->samples = samples;
	ann_layer->neurons_previous = neurons_previous;
	ann_layer->neurons = neurons;
	//W
	ann_layer->W = initRandomMatrix(neurons_previous, neurons);
	//B
	ann_layer->B = initB(samples, neurons);
	//Z
	ann_layer->Z = initZeroMatrix(samples, neurons);
	//A
	ann_layer->A = initZeroMatrix(samples, neurons);
	//dZ
	ann_layer->dZ = initZeroMatrix(samples, neurons);
	//dW
	ann_layer->dW = initZeroMatrix(neurons_previous, neurons);
	//dB
	ann_layer->dB = initZeroMatrix(samples, neurons);
	//Import the layer type and the activation
	ann_layer->layer_type = layer_type;
	ann_layer->activation = activation;
	//Return the initialized ANNLayer
	return ann_layer;
}

//Method to dispose an ANNLayer
void disposeANNLayer(ANNLayer* ann_layer)
{
	//Dispose the matrices of the layer
	matrixDispose(ann_layer->W, ann_layer->neurons_previous);
	matrixDispose(ann_layer->B, ann_layer->samples);
	matrixDispose(ann_layer->Z, ann_layer->samples);
	matrixDispose(ann_layer->A, ann_layer->samples);
	matrixDispose(ann_layer->dZ, ann_layer->samples);
	matrixDispose(ann_layer->dW, ann_layer->neurons_previous);
	matrixDispose(ann_layer->dB, ann_layer->samples);
	//Dispose the ANNLayer itself
	free(ann_layer);
	ann_layer = NULL;
}

