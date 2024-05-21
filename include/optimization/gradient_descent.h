//Gradient descent class of LibBQsC by Berkay

/**
 * Note : 	Instances of this class should be initialized using the constructor
 * 			method provided. Initialization of GradientDescent instances with
 * 			braces may lead to undefined behavior or incomplete initialization.
 *
 * Note : 	This class utilizes vectors to maintain compatibility with algorithms
 * 			that expect weights in vector format. Therefore, weight matrices
 * 			should be flattened into vectors.
 */

#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

/**
 * GradientDescent structure
 */
typedef struct
{
	//Weight vector w and its size
	double* w;
	int n;
	//Learning rate
	double learning_rate;
}
GradientDescent;

/**
 * Constructor method of the gradient descent class
 *
 * @param 	w	weight vector
 * @param 	n 	size of the weight vector
 * @return		pointer to the initialized GradientDescent
 */
GradientDescent* initGradientDescent(double* w, int n);

/**
 * Method to update the weights
 *
 * First checks if the passed gradient is valid, then updates the
 * weight vector w. Returns the updated w too.
 *
 * @param	gradientDescent		the GradientDescent
 * @param	gradient			gradient of the weight vector
 * @param	n					size of the gradient
 * @return 						updated weight vector
 */
double* updateGradientDescent(GradientDescent* gradientDescent, double* gradient, int n);

/**
 * Method to dispose a GradientDescent
 *
 * @param	gradientDescent		GradientDescent to be disposed
 * @param	dispose_w			1 if the w will be disposed
 */
void disposeGradientDescent(GradientDescent* gradientDescent, int dispose_w);

#endif //GRADIENT_DESCENT_H
