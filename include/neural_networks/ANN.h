//ANN class of LibBQsC by Berkay

/**
 * Note : 	Some of the most important methods in this class are static thus are not shown
 * 			in this header file.
 */

#ifndef ANN_H
#define ANN_H

#include "neural_network_utilities.h"

/**
 * ANN struct
 */
typedef struct
{
	//Input data
	double** X;
	double** Y;
	//Input data dimensions
	int samples;
	int features;
	int classes;
	//Layers of the ANN
	ANNLayer** layers;
	int number_of_layers;
}
ANN;

/**
 * Method to initialize an ANN
 *
 * @param X			X input data
 * @param Y			labels of the X
 * @param samples	number of samples in the X and Y
 * @param features	number of features in the X
 * @param classes	number of classes in the Y
 * @return			pointer to the initialized ANN
 */
ANN* initANN(double** X, double** Y, int samples, int features, int classes);

/**
 * Method to add a layer to the ANN
 *
 * @param ann			ANN to which the hidden layer will be added
 * @param neurons		number of neurons to be in the hidden layer
 * @param layer_type	type of the layer to be added
 * @param activation	activation function to be used in the hidden layer
 */
void addLayerANN(ANN* ann, int neurons, LayerType layer_type, Activation activation);

/**
 * Method to train the ANN
 *
 * @param ann				ANN to be trained
 * @param max_iterations	maximum number of iteration
 * @param threshold			tolerance to be used to check converge
 */
void trainANN(ANN* ann, int max_iterations, double threshold);

/**
 * Method to make a prediction
 *
 * @param ann		trained ANN
 * @param X			data points to be predicted
 * @param samples	number of data points in the X to be predicted
 * @param features	number of features in the X to be predicted
 * @return			activated output layer of the ANN
 */
double** predictANN(ANN* ann, double** X, int samples, int features);

/**
 * Method to dispose an ANN
 *
 * @param ann	ANN to be disposed
 */
void disposeANN(ANN* ann);

#endif //ANN_H
