//Optimization config class of LibBQsC by Berkay

#include "../../include/optimization/optimization_config.h"

//Gradient descent parameters
double gradient_descent_learning_rate = 0.001;

//ADAM optimizer parameters
double adam_learning_rate = 0.001;
double adam_beta_1 = 0.9;
double adam_beta_2 = 0.999;
double adam_epsilon = 1e-8;
