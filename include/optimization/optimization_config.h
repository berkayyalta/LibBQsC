//Optimization config class of LibBQsC by Berkay

#ifndef OPTIMIZATION_CONFIG_H
#define OPTIMIZATION_CONFIG_H

/**
 * Optimizer enum
 */
typedef enum
{
	GRADIENT_DESCENT,
	ADAM_OPTIMIZER
}
Optimizer;

//Gradient descent parameters
extern double gradient_descent_learning_rate;

//ADAM optimizer parameters
extern double adam_learning_rate;
extern double adam_beta_1;
extern double adam_beta_2;
extern double adam_epsilon;

#endif //OPTIMIZATION_CONFIG_H
