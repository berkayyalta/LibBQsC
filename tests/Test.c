#include <stdio.h>
#include <stdlib.h>

#include "../include/libBQsC.h"

#include "../tests/sample_data.h"

int main()
{
	//Import the X and Y data
	double** X = getX();
	double** Y = getY();

	/*
	 * Sample use of the ANN
	 */

	//Initilize	 the ANN
	ANN* ann = initANN(X, Y, 500, 2, 1);
	//Add the hidden layers of the ANN
	addLayerANN(ann, 12, HIDDEN_LAYER, SIGMOID);
	addLayerANN(ann, 6, HIDDEN_LAYER, SIGMOID);
	addLayerANN(ann, 1, OUTPUT_LAYER, SIGMOID);
	//Train the ANN
	trainANN(ann, 5000, 1e-8);
	//Initialize a pointer to predict
	double** pointer_to_predict = initMatrix(1, 2);
	pointer_to_predict[0][0] = 1.5;
	pointer_to_predict[0][1] = 0.5;
	//Make the prediction
	double** result = predictANN(ann, pointer_to_predict, 1, 2);
	//Print the result
	printMatrix(result, 1, 1, 16);
	//Dispose the ANN
	disposeANN(ann);

	/*
	 * Sample use of the LogisticRegression
	 */

	//Set debugTrainingLogisticRegression to 1 so t and loss will be printed at evey time step
	debugTrainingLogisticRegression = 1;
	//Initialize the LogisticRegression
	LogisticRegression* regr = initLogisticRegression(X, Y, 500, 2, 1);
	//Train the LogisticRegression
	trainLogisticRegression(regr, ADAM_OPTIMIZER, 5000, 1e-8);
	//Print the logistic regression
	printf("\n");
	printLogisticRegression(regr, 3);
	//Dispose the logistic regression
	disposeLogisticRegression(regr);

	//Exit success
	return EXIT_SUCCESS;
}
