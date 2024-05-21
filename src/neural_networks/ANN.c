//ANN class of LibBQsC by Berkay

#include "../../include/neural_networks/ANN.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../include/core/linear_algebra.h"
#include "../../include/metrics/regression_metrics.h"
#include "../../include/optimization/adam_optimizer.h"

//Method to initialize an ANN
ANN* initANN(double** X, double** Y, int samples, int features, int classes)
{
	//Initialize the ANN and handle any allocation failure
	ANN* ann = malloc(sizeof(ANN));
	if (ann == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Import the X and Y
	ann->X = X;
	ann->Y = Y;
	//Input data dimensions
	ann->samples = samples;
	ann->features = features;
	ann->classes = classes;
	//Layers of the ANN
	ann->layers = NULL;
	ann->number_of_layers = 0;
	//Return the initialized ANN
	return ann;
}

/**
 * This method increses the size of the ann->layers by generating a new array, copying the items in it and disposing
 * the old one.
 */

//Method to add a layer to the ANN
void addLayerANN(ANN* ann, int neurons, LayerType layer_type, Activation activation)
{
	//Initialize the new ann->layers with the size of ann->number_of_layers + 1 and handle any allocation failure
	ANNLayer** new_layer_array = malloc((ann->number_of_layers+1) * sizeof(ANNLayer*));
	if (new_layer_array == NULL)
	{
		printf("Failed to allocate memory");
		exit(EXIT_FAILURE);
	}
	//Copy the existing layers into the new array
	for (int i = 0; i < ann->number_of_layers; i++)
	{
		new_layer_array[i] = ann->layers[i];
	}
	//Dispose the old array
	free(ann->layers);
	ann->layers = new_layer_array;
	//Decide the number of neurons in the previous layer and initialize the new layer
	int neurons_previous = (ann->number_of_layers == 0) ? ann->features : ann->layers[ann->number_of_layers-1]->neurons;
	ANNLayer* new_layer = initANNLayer(ann->samples, neurons_previous, neurons, layer_type, activation);
	//Add the new layer to the ANN and increase the ann->number_of_layers by one
	ann->layers[ann->number_of_layers] = new_layer;
	ann->number_of_layers += 1;
}

/**
 * This method performs the Z = XW + B and A = activation_function(Z) operations for the layer with the specified
 * index element-wise. It is "update" because it doesn't calculates and returns something but rather updates the
 * matrices of the ANNLayers of the ANN.
 */

//Method to update the outputs of a layer
static void updateLayerOutputs(ANN* ann, int layer_no)
{
	//Iterate over the rows of the X/A[l-1] (also the rows of the Z[l] and A[l]) to perform the XW
	for (int row_no = 0; row_no < ann->samples; row_no++)
	{
		//Iterate over the columns of the Wi (also the columns of the Z[l] and A[l]) to perform the XW
		for (int column_no = 0; column_no < ann->layers[layer_no]->neurons; column_no++)
		{
			//Iterate to calculate the result of the matrix multiplication
			double result_XW = 0.0;
			for (int item_no = 0; item_no < ann->layers[layer_no]->neurons_previous; item_no++)
			{
				//Use X if this is the first layer
				if (layer_no == 0)
				{
					result_XW += ann->X[row_no][item_no] * ann->layers[layer_no]->W[item_no][column_no];
				}
				//Use A[l-1] otherwise
				else
				{
					result_XW += ann->layers[layer_no-1]->A[row_no][item_no] * ann->layers[layer_no]->W[item_no][column_no];
				}
			}
			//Z[l][i][j] = XW[l][i][j] + B[l][i][j]
			ann->layers[layer_no]->Z[row_no][column_no] = result_XW + ann->layers[layer_no]->B[row_no][column_no];
			//A[l][i][j] = activation_function(Z[l][i][j])
			ann->layers[layer_no]->A[row_no][column_no] = activationFunction(ann->layers[layer_no]->Z[row_no][column_no], ann->layers[layer_no]->activation);
		}
	}
	//Z and A matrices of the current layer are now updated so the next layer (l+1) can be calculated using the A of the current layer
}

//Method to perform the complete forward propagation
static void forwardPropagationANN(ANN* ann)
{
	//Iterate over the layers
	for (int layer_no = 0; layer_no < ann->number_of_layers; layer_no++)
	{
		//Update the layers using the method
		updateLayerOutputs(ann, layer_no);
	}
	//ann->layers[ann->number_of_layers-1]->A = is the output layer of the ANN
}

/**
 * These three methods are to be used in the backward_propagation method to update the gradients of the layers.
 *
 * dZ[L] = A[L] - Y						and 	dZ[l] = (dZ[l+1] x W[l+1]^T) * activation_function_derivative(Z[l])
 * dW[l] = 1/m * (A[l-1]^T x dZ[l]) 	and 	dW[l=1] = 1/m * (X^T x dZ[l])
 * dB[l] = mean(dZ[l])
 */

//Method to update the dZ of a layer
static double* update_dZ(ANN* ann, int layer_no)
{
	//Generate the vector to be the rows of the dB
	double* db = initZeroVector(ann->layers[layer_no]->neurons);
	//Iterate over the rows of the dZ (samples, neurons) to update it
	for (int row_no = 0; row_no < ann->samples; row_no++)
	{
		//Iterate over the columns of the dZ (samples, neurons) to update it
		for (int column_no = 0; column_no < ann->layers[layer_no]->neurons; column_no++)
		{
			//Calculate the dZ for the output layer : dZ[L] = A[L] - Y
			if (layer_no == ann->number_of_layers-1)
			{
				ann->layers[layer_no]->dZ[row_no][column_no] = ann->layers[layer_no]->A[row_no][column_no] - ann->Y[row_no][column_no];
			}
			//Calculate the dZ for the hidden layers : dZ[l] = (dZ[l+1] x W[l+1]^T) * activation_function_derivative(Z[l])
			else
			{
				//Calculate the current item of dZ[l+1] x W[l+1]^T (which dZlp1xWlp1T stands for) by performing the matrix multiplication
				double dZlp1xWlp1T = 0.0;
				for (int item_no = 0; item_no < ann->layers[layer_no+1]->neurons; item_no++)
				{
					dZlp1xWlp1T += ann->layers[layer_no+1]->dZ[row_no][item_no] * ann->layers[layer_no+1]->W[column_no][item_no];
				}
				//Update the current item of dZ by doing dZlp1xWlp1T * activation_function_derivative(Z[l][i][j])
				ann->layers[layer_no]->dZ[row_no][column_no] = dZlp1xWlp1T * activationFunctionDerivative(ann->layers[layer_no]->Z[row_no][column_no], ann->layers[layer_no]->activation);
			}
			//A column of db is the mean of the column of dZ and dB will be generated by tiling the db
			db[column_no] += ann->layers[layer_no]->dZ[row_no][column_no];
		}
	}
	//Divide the db by samples to calculate the means then return the it to generate the dB
	for (int db_i = 0; db_i < ann->layers[layer_no]->neurons; db_i++)
	{
		db[db_i] /= ann->samples;
	}
	return db;
	//dZ matrix of the current layer is now updated so the next layer (l-1) can be calculated using the dZ of the current layer
}

//Method to update the dW of a layer
static void update_dW(ANN* ann, int layer_no)
{
	//Iterate over the rows of the dW (neurons_previous, neurons) to update it
	for (int row_no = 0; row_no < ann->layers[layer_no]->neurons_previous; row_no++)
	{
		//Iterate over the columns of the dW (neurons_previous, neurons) to update it
		for (int column_no = 0; column_no < ann->layers[layer_no]->neurons; column_no++)
		{
			//Calculate the dW for the hidden layers in between : 1/m * (A[l-1]^T x dZ[l])
			if (layer_no > 0)
			{
				//Calculate the current item of (A[l-1]^T x dZ[l]) (which Alm1TxdZl stands for)
				double Alm1TxdZl = 0.0;
				for (int item_no = 0; item_no < ann->samples; item_no++)
				{
					Alm1TxdZl += ann->layers[layer_no-1]->A[item_no][row_no] * ann->layers[layer_no]->dZ[item_no][column_no];
				}
				//Update the current item of dw
				ann->layers[layer_no]->dW[row_no][column_no] = (1.0/ann->samples) * Alm1TxdZl;
			}
			//Calculate the dW for the first hidden layer (l=1) : 1/m * (X^T x dZ[l])
			else if (layer_no == 0)
			{
				//Calculate the current item of (X^T x dZ[l]) (which XTxdZl stands for)
				double XTxdZl = 0.0;
				for (int item_no = 0; item_no < ann->samples; item_no++)
				{
					XTxdZl += ann->X[item_no][row_no] * ann->layers[layer_no]->dZ[item_no][column_no];
				}
				//Update the current item of dw
				ann->layers[layer_no]->dW[row_no][column_no] = (1.0/ann->samples) * XTxdZl;
			}
		}
	}
	//dW matrix of the current layer is now updated
}

//Method to update the dB of a layer
static void update_dB(ANN* ann, double* db, int layer_no)
{
	//Iterate over the rows of the dB (samples, neurons) to update it
	for (int row_no = 0; row_no < ann->samples; row_no++)
	{
		//Iterate over the columns of the dB (samples, neurons) to update it
		for (int column_no = 0; column_no < ann->layers[layer_no]->neurons; column_no++)
		{
			//Update the current item of dB
			ann->layers[layer_no]->dB[row_no][column_no] = db[column_no];
		}
	}
}

//Method to perform the complete backward propagation
static void backwardPropagationANN(ANN* ann)
{
	//Iterate over the layers starting from the output layer
	for (int layer_no = ann->number_of_layers-1; layer_no > -1; layer_no--)
	{
		//Update the dZ of the current layer
		double* db = update_dZ(ann, layer_no);
		//Update the dW of the current layer
		update_dW(ann, layer_no);
		//Update the dB of the current layer and dispose the db
		update_dB(ann, db, layer_no);
		free(db);
		db = NULL;
	}
	//Gradients of each layer are now updated
}

//Method to train the ANN
void trainANN(ANN* ann, int max_iterations, double threshold)
{
	//Check if the ANN is initialized appropriately having an output layer
	if (ann->layers[ann->number_of_layers-1]->layer_type == OUTPUT_LAYER)
	{
		//Initialize the arrays of ADAMs
		ADAM* adamW[ann->number_of_layers];
		ADAM* adamB[ann->number_of_layers];
		for (int layer_no = 0; layer_no < ann->number_of_layers; layer_no++)
		{
			//Initialize an ADAM for the current W
			double* current_flattened_W = flatten(ann->layers[layer_no]->W, ann->layers[layer_no]->neurons_previous, ann->layers[layer_no]->neurons, 0);
			adamW[layer_no] = initADAM(current_flattened_W, ann->layers[layer_no]->neurons_previous * ann->layers[layer_no]->neurons);
			//Initialize an ADAM for the current B
			double* current_flattened_B = flatten(ann->layers[layer_no]->B, ann->samples, ann->layers[layer_no]->neurons, 0);
			adamB[layer_no] = initADAM(current_flattened_B, ann->samples * ann->layers[layer_no]->neurons);
		}
		//Initialize the previous loss as INT_MAX
		double loss_previous = INT_MAX;
		//Begin the iteration
		for (int t = 0; t < max_iterations; t++)
		{
			//Perform the propagations and update the matrices
			forwardPropagationANN(ann);
			backwardPropagationANN(ann);
			//Update the optimizers and the matrices
			for (int layer_no = 0; layer_no < ann->number_of_layers; layer_no++)
			{
				//UPDATE THE W MATRIX
				//Flatten the gradient into dw
				double* dw = flatten(ann->layers[layer_no]->dW, ann->layers[layer_no]->neurons_previous, ann->layers[layer_no]->neurons, 0);
				//Update the optimizer
				double* updated_w = updateADAM(adamW[layer_no], dw, (ann->layers[layer_no]->neurons_previous * ann->layers[layer_no]->neurons));
				//Dispose the flattened gradient
				free(dw);
				//Update the W matrix of the current layer out of the w vector returned from the optimizer
				int index_w = 0;
				for (int row_no = 0; row_no < ann->layers[layer_no]->neurons_previous; row_no++)
				{
					for (int column_no = 0; column_no < ann->layers[layer_no]->neurons; column_no++)
					{
						ann->layers[layer_no]->W[row_no][column_no] = updated_w[index_w];
						index_w += 1;
					}
				}
				//UPDATE THE B MATRIX
				//Flatten the gradient into db
				double* db = flatten(ann->layers[layer_no]->dB, ann->samples, ann->layers[layer_no]->neurons, 0);
				//Update the optimizer
				double* updated_b = updateADAM(adamB[layer_no], db, (ann->samples * ann->layers[layer_no]->neurons));
				//Dispose the flattened gradient
				free(db);
				//Update the B matrix of the current layer out of the b vector returned from the optimizer
				int index_b = 0;
				for (int row_no = 0; row_no < ann->samples; row_no++)
				{
					for (int column_no = 0; column_no < ann->layers[layer_no]->neurons; column_no++)
					{
						ann->layers[layer_no]->B[row_no][column_no] = updated_b[index_b];
						index_b += 1;
					}
				}
			}








			//Loss calculation will be here
			if (t%1 == 0)
			{
				//Calculate the current loss
				double* y_true = flatten(ann->Y, ann->samples, ann->classes, 0);
				double* y_pred = flatten(ann->layers[ann->number_of_layers-1]->A, ann->samples, ann->classes, 0);
				double current_loss = logLoss(y_true, y_pred, ann->samples * ann->classes);
				//Break if necessary
				if (((loss_previous - current_loss) < threshold) && (t > 1000))
				{
					printf("BREAK\n");
					break;
				}
				//Update the previous loss
				loss_previous = current_loss;
			}
			printf("Iteration : %d, Loss : %lf\n", t, loss_previous);








		}
		//Dispose the optimizers after the optimization
		for (int layer_no = 0; layer_no < ann->number_of_layers; layer_no++)
		{
			//Dispose for the current W
			disposeADAM(adamW[layer_no], 1);
			//Dispose for the current W
			disposeADAM(adamB[layer_no], 1);
		}
	}
	//Throw exception otherwise
	else
	{
		printf("The ANN does not have an output layer");
		exit(EXIT_FAILURE);
	}
}

//Method to make a prediction
double** predictANN(ANN* ann, double** X, int samples, int features)
{
	//Declare the A, which will become the result, as X, which is the input layer
	double** A = X;
	//Iterate over the layers
	for (int layer_no = 0; layer_no < ann->number_of_layers; layer_no++)
	{
		//If the previous A is not the input layer X, dispose it
		if (layer_no != 0)
		{
			matrixDispose(A, samples);
		}
		/*
		 * Calculate the A of the current layer (A_l) then update the A
		 */
		//Initialize the A of the current layer to be returned
		double** A_l = initMatrix(samples, ann->layers[layer_no]->neurons);
		//Iterate over the rows of the output of the previous layer
		for (int row_no = 0; row_no < samples; row_no++)
		{
			//Iterate ove the columns of the W of the current layer
			for (int column_no = 0; column_no < ann->layers[layer_no]->neurons; column_no++)
			{
				//Initialize the current item of the XW as 0
				double XW = 0.0;
				//Iterate over the rows of the W of the current layer to perform the matrix multiplication
				for (int item_no = 0; item_no < ann->layers[layer_no]->neurons_previous; item_no++)
				{
					XW += A[row_no][item_no] * ann->layers[layer_no]->W[item_no][column_no];
				}
				//Calculate the current item of the A_l
				A_l[row_no][column_no] = activationFunction((XW + ann->layers[layer_no]->B[0][column_no]), ann->layers[layer_no]->activation);
			}
		}
		//Update the A
		A = A_l;
	}
	//Return the A
	return A;
}

//Method to dispose an ANN
void disposeANN(ANN* ann)
{
	//Dispose the layers
	for (int layer_no = 0; layer_no < ann->number_of_layers; layer_no++)
	{
		disposeANNLayer(ann->layers[layer_no]);
	}
	//Dispose the ANN
	free(ann);
	ann = NULL;
}

