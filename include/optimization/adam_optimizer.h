//ADAM optimizer class of LibBQsC by Berkay

/**
 * Note : 	Instances of this class should be initialized using the constructor
 * 			method provided. Initialization of ADAM instances with braces may
 * 			lead to undefined behavior or incomplete initialization.
 *
 * Note : 	This class utilizes vectors to maintain compatibility with algorithms
 * 			that expect weights in vector format. Therefore, weight matrices
 * 			should be flattened into vectors.
 */

#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

/**
 * ADAM structure
 */
typedef struct
{
	//Weight vector w and the moment estimates m and v
	double* w;
	double* m;
	double* v;
	//Size of the vectors
	int n;
	//Constant variables
	double learning_rate;
	double beta_1;
	double beta_2;
	double epsilon;
	//Time step t
	int t;
}
ADAM;

/**
 * Constructor method of the ADAM optimizer class
 *
 * Initializes the ADAM optimizer along with its moment estimates
 *
 * @param 	w	weight vector
 * @param 	n 	size of the weight vector
 * @return		pointer to the initialized ADAM optimizer
 */
ADAM* initADAM(double* w, int n);

/**
 * Method to update the weights
 *
 * First checks if the passed gradient is valid, then updates the
 * weight vector w of the passed ADAM using the ADAM update rule.
 * Returns the updated w too.
 *
 * @param	adam		the ADAM
 * @param	gradient	gradient of the weight vector
 * @param	n			size of the gradient
 * @return 				updated weight vector
 */
double* updateADAM(ADAM* adam, double* gradient, int n);

/**
 * Method to dispose an ADAM optimizer
 *
 * @param	adam		ADAM to be disposed
 * @param	dispose_w	1 if the w will be disposed
 */
void disposeADAM(ADAM* adam, int dispose_w);

#endif //ADAM_OPTIMIZER
