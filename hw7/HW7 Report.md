# HW7 Report

> Author: Jiayi Liu (9749111299)
>
> This is an individual work.

`python hmm.py` can rerun the code.

## Implementation

* All the implementation is in the class `HMM`. 

* A recursive process was used to infer the possible hidden states step by step.
* In each recursion, the accumulated possibility so far, the trace of transition states and the index of the current time step will be passed. 
* A valid emission will invoke further recursions.
* After looking through all the obvervations, the recursion will terminate. All the valid squences reaching the end will be stored together with the corresponding possibility.
* Finally, we choose the final sequence with the maximum possibility.

## Results

The output from the program is as follows.

```javascript
The Most Likely Sequence is: 
[7, 6, 5, 6, 5, 4, 5, 6, 7, 8]
The maximum probability is:  6.510416666666667e-05
```



