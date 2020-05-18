# Decision Tree From Scratch
Decision Tree Algorithm written in Python with Numpy and Pandas.
## 1. Overview of the Implemention
The Decision Tree algorithm implemented here can accommodate customisations in the maximum decision tree depth, the minimum sample size, the number of random features if the users want to choose randomly some `d` features without replacement when splitting a node, and the number of random splits if the users want to split a node for some `s` times and choose the best split among these `s` splits instead of choosing the best split among all potential splits of the node.

The algorithm is not able to work with datasets containing categorical data natively, so it requires those datasets to be preprocessed such as converting ordinal data into integers. After converting all categorical string values in a dataset to integers, a user can use that dataset with the algorithm.
## 2. Testing Specifications
- The depth of the decision tree starts at 1 for the first time and increases by 1 until the accuracy rate on the training dataset reaches 100%.
- NO RANDOMNESS. A node is split at its best split, and all features of the dataset will be considered when determining a node’s best split from all of its potential splits.
- There is no preprocessing required for the UCI Breast Cancer Wisconsin (Diagnostic) Dataset.
- The UCI Car Evalution Dataset will be preprocessed as follows.
```
"buying": "low" -> 1, "med" -> 2, "high" -> 3, "vhigh" -> 4
"maint": "low" -> 1, "med" -> 2, "high" -> 3, "vhigh" -> 4
"doors": "2" -> 2, "3" -> 3, "4" -> 4, "5more" -> 5
"persons": "2" -> 2, "4" -> 4, "more" -> 6
"lug_boot": "small" -> 1, "med" -> 2, "big" -> 3
"safety": "low" -> 1, "med" -> 2, "high" -> 3
```
## 3. Results on UCI Breast Cancer Wisconsin (Diagnostic) Dataset
Overfitting can be observed when the depth of the decision tree changes from 3 to 4, in which the testing accuracy decreases from 95.10% to 94.41% while the training accuracy increases from 97.65% to 98.36%.
```
maxDepth = 1: accTest = 88.11%, accTrain = 92.25%, buildTime = 1.36s
maxDepth = 2: accTest = 90.21%, accTrain = 92.72%, buildTime = 2.49s
maxDepth = 3: accTest = 95.10%, accTrain = 97.65%, buildTime = 3.35s
maxDepth = 4: accTest = 94.41%, accTrain = 98.36%, buildTime = 4.09s
maxDepth = 5: accTest = 95.80%, accTrain = 99.30%, buildTime = 4.23s
maxDepth = 6: accTest = 95.80%, accTrain = 99.77%, buildTime = 4.09s
maxDepth = 7: accTest = 95.80%, accTrain = 100.00%, buildTime = 4.04s
```
## 4. Results on UCI Car Evaluation Dataset
Overfitting can be observed when the depth of the decision tree changes from 6 to 7, in which the testing accuracy decreases from 93.26% to 91.91% while the training accuracy increases from 93.55% to 94.54%.
```
maxDepth = 1: accTest = 68.98%, accTrain = 70.47%, buildTime = 0.01s
maxDepth = 2: accTest = 79.19%, accTrain = 77.17%, buildTime = 0.02s
maxDepth = 3: accTest = 79.19%, accTrain = 79.16%, buildTime = 0.02s
maxDepth = 4: accTest = 83.62%, accTrain = 85.69%, buildTime = 0.03s
maxDepth = 5: accTest = 86.90%, accTrain = 87.26%, buildTime = 0.03s
maxDepth = 6: accTest = 93.26%, accTrain = 93.55%, buildTime = 0.05s
maxDepth = 7: accTest = 91.91%, accTrain = 94.54%, buildTime = 0.06s
maxDepth = 8: accTest = 95.57%, accTrain = 98.35%, buildTime = 0.07s
maxDepth = 9: accTest = 95.76%, accTrain = 99.09%, buildTime = 0.07s
maxDepth = 10: accTest = 97.30%, accTrain = 99.83%, buildTime = 0.08s
maxDepth = 11: accTest = 97.30%, accTrain = 100.00%, buildTime = 0.08s
```
## 5. References
1. [Sebastian Mantey's repository](https://github.com/SebastianMantey/Decision-Tree-from-Scratch)
2. [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
3. [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
