//Neural network utilities class of LibBQsC by Berkay

/**
 * Note : 	This class has some utilities that are expected to be used across multiple ANN and ANN related
 * 			classes.
 */

#ifndef NEURAL_NETWORK_UTILITIES_H
#define NEURAL_NETWORK_UTILITIES_H

/**
 * LayerType enum
 *
 * Each layer will have a type to check if the model is initialized appropriately having
 * an output layer.
 */
typedef enum
{
	HIDDEN_LAYER,
	OUTPUT_LAYER
}
LayerType;

/**
 * Activation enum
 *
 * Each layer will be activated using the activation function specified for that layer
 */
typedef enum
{
	RELU,
	SIGMOID,
	TANH
}
Activation;

/**
 * Method to apply the activation function
 *
 * @param z				double to be plugged into the activation function
 * @param activation	activation function to be applied
 * @return				result of the activation function
 */
double activationFunction(double z, Activation activation);

/**
 * Method to apply the derivative of the activation function
 *
 * @param z				double to be plugged into the activation function's derivative
 * @param activation	activation function to be applied
 * @return				result of the activation function's derivative
 */
double activationFunctionDerivative(double z, Activation activation);

/**
 * ANNLayer struct
 *
 * Layer struct to organize the ANN sturct
 */
typedef struct
{
	//Dimensions of the matrices
	int samples;
	int neurons_previous;
	int neurons;
	//Weight matrix W : (neurons in the previous layer, neurons in this layer)
	double** W;
	//Intercept matrix B : (samples, neurons in this layer)
	double** B;
	//Weighted sum matrix Z : (samples, neurons in this layer)
	double** Z;
	//Activation matrix A : (samples, neurons in this layer)
	double** A;
	//dL/dZ : (samples, neurons in this layer)
	double** dZ;
	//dL/dW : (neurons in the previous layer, neurons in this layer)
	double** dW;
	//dL/dB : (samples, neurons in this layer)
	double** dB;
	//Type and the activation of the layer
	LayerType layer_type;
	Activation activation;
}
ANNLayer;

/**
 * Method to initialze an ANNLayer
 *
 * First initializes the ANNLayer then initializes its matrices with the required dimensions
 *
 * @param	samples				number of data points in the X
 * @param	neurons_previous	number of neurons in the previous layer
 * @param 	neurons				number of neurons to be in this layer
 * @param	layer_type			type of the layer
 * @param	activation			activation of the layer
 * @return						pointer to the initialized ANNLayer
 */
ANNLayer* initANNLayer(int samples, int neurons_previous, int neurons, LayerType layer_type, Activation activation);

/**
 * Method to dispose an ANNLayer
 *
 * First disposes the matrices of the ANNLayer then disposes the ANNLayer itself
 *
 * @param ann_layer		ANNLayer to be disposed
 */
void disposeANNLayer(ANNLayer* ann_layer);

#endif //NEURAL_NETWORK_UTILITIES_H
