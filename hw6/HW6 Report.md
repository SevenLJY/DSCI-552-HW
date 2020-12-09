# HW6 Report

> Author: Jiayi Liu (9749111299)
>
> This is an individual work.

This project structure is as follows.

```python
# Implementation of SVM for linear separable data
svm.py
# Implementation of SVM with kernel function
svm_kernel.py
```

## Part (a): Linear Data

`python svm.py` can rerun the code. 

The implementation of quadratic programming part in this implementation is with the help of `cvxopt` library and the "matrix" data struction in it. 

When filtering the support vectors, threshold got set as`1e-6` since most of the results are extremely close to zero but not zero. 

The output is as follows. 

```python
W: [ 7.2500563  -3.86188924]
b:  [-0.10698734]
Support Vectors: 
[[0.24979414 0.18230306]
 [0.3917889  0.96675591]
 [0.02066458 0.27003158]]
```

So the fattest margin line is `7.2500563 * x - 3.86188924 * y - 0.10698734 = 0`.

## Part (b): Non-linear Data

`python svm_kernel.py` can rerun the code.

The polynomial kernel function K = (1 + x<sup>T</sup>x)<sup>2</sup> is used in this implementation. The quadratic programming part stays the same as above.

The output is as follows.

```python
W: [ 1.71807848e-15  9.14594548e-01 -9.32813694e-01  4.88203726e+00
 -2.29832155e+00  6.93588632e-01]
b:  [-0.20283088]
Support Vectors: 
[[0.24979414 0.18230306]
 [0.22068726 0.11139981]
 [0.3917889  0.96675591]
 [0.02066458 0.27003158]]
```

So the curve equation is [ 1.71807848e-15  9.14594548e-01 -9.32813694e-01  4.88203726e+00 -2.29832155e+00  6.93588632e-01]<sup>T</sup> z - 0.20283088 = 0.

