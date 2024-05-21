
# LibBQsC

## Overview

LibBQsC is a machine learning library that has implementations that aim to minimize the computational costs at the first place. Currently, it has ANN and logistic regression implementations along with some other helper classes. However, many other machine learning algorithms and features will be added in the next versions of the library.

## Classes
- **Linear Algebra** : Linear algebra class consists of methods related to vectorized operations. For now, some of these methods are not utilized since the methods involving extensive computations have their own implementations that do the operations with a single iteration. The other methods such as `initMatrix()` are effectively used throughout the library. 

---

- **Regression Metrics** : Regression metrics class has implementations for common loss functions *MSE*, *MAE*, and *log loss*. These implementations are to evaluate a model rather than to be minimized to train a model.

---

- **ANN** : ANN class is an artificial neural network implementation. It works similar to SKLearn's *MLPC*, but it has activation function diversity differently. Some other important features are planned to be added to this class in the next versions of the library.

- **Neural Network Utilities** : This class has structs and methods that are used in the ANN class and are likely to be used in the other neural network models that are planned to be implemented.

---

- **ADAM Optimizer** : ADAM optimizer is the defult optimizer of the implemented classes and the most powerful optimizer in this library for now. 

- **Gradient Descent** : This is a gradient descent implementation to provide an alternative optimization algrotihm. Some other optimization algorithms as well are planned to be added in the next versions of the library.

---

- **Feature Scaling** : Feature scaling class has implementations for *min-max scaling* and *standardizing*.

---

- **Logistic Regression** : This class is a logistic regression implementation that can do both binomial and multinomial classification.

---

- **Statistics** : Statistics class has implementations for calculating fundamental Statistics such as mean and standard deviation of a data.

## Usage

Methods in this library are designed to be as easy to use as possible. In the provided file `Test.c`, there are example uses of ANN and logistic regression classes. The future implementations will follow a similar format.

## Planned changes

There are a lot of features that are planned to be added in the next versions of the library. Some of them are listed in order of priority below : 

- A linear regression class will be implemented.

- ANN Changes
    - Bias vectors will modified to be one dimensional and the required changes will be made on the related methods.
    - *Softmax* function and multinomial classification will be implemented.
    - Batch processing will be implemented.
    - A better converge criteria will be implemented.
    - A method to return a prediction made by an ANN as labels rather than raw numbers will be implemented.

- A class to import CSV data will be implemented.

- Some other optimization algorithms will be implemented.

- Some model evaluation algorithms such as k-fold and sequential cross validations will be added.

- SVR and SVC are planned to be implemented.

## License

This project is licensed under the GNU General Public License version 2.0 (GPL-2.0) - see the LICENSE file for details.
