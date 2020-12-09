# HW4 Report

> Author: Jiayi Liu (9749111299)

This is an individual work. The project structure is as follows.

```python
# Perceptron Learning Implementation
perceptron_learning.py
# Pocket Learning.py Implementation
pocket_learning.py
# Logistic Regression Implementation
logistic_regression.py
# Linear Regression Implementation
linear_regression.py
```

## Perceptron Learning

Command `python perceptron_learning.py` will rerun the code. All the implementation is inside the class `Perceptron`. Since the weights is randomly generated, the convergence iteration is unpredictable. So I set up the maximum number of iterations as 10000. The termination condition is either convergence or reaching maximum iterations. The weights update was implemented in a loop. And the learning rate is assumed as `1e-4`. Here is the result from one of the experiments.

```python
Weight: 
[[ 2.64562809e-01]
 [-2.12239513e-01]
 [-1.58798357e-01]
 [ 2.26250376e-04]]
Iteration:  396
Accuracy 1.0
```

## Pocket Algorithm

Command `python pocket_learning.py` will rerun the code. All the implementation is inside the class `Pocket`. The weights update was implemented in a loop. The learning rate is assumed as `1e-4`. Here is the result from one of the experiments.

```python
Weights: 
[[-0.03735273]
 [ 0.06602827]
 [-0.00565734]
 [-0.02728441]]
Accuracy:  0.517
```

Here is the plot of the number of (minimum) misclassification points against the iterations.

<img src="/Users/ljiayi/Documents/USC/DSCI 552/hw4/pocket.png" alt="pocket" style="zoom:60%;" />

## Logistic Regression

Command `python logistic_regression.py` will rerun the code. All the implementation is inside the class `LogisticReg`. The learning rate is assumed as `1e-4`. Here is the result from one of the experiments.

```python
Weight: 
[[0.81976551]
 [0.44200576]
 [0.16682609]
 [0.43772866]]
Accuracy:  0.494
```

## Linear Regression

Command `python linear_regression.py` will rerun the code. All the implementation is inside the class `LinearReg`. This algorithm is a one-shot solution, so only one iteration is enough. Here are the optimal weights.

```python
[[1.08546357]
 [3.99068855]
 [0.01523535]]
```

