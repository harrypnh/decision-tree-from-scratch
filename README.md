# Decision Tree From Scratch
Decision Tree Algorithm written in Python using NumPy and Pandas.
## 1. Overview of the Implemention
The Decision Tree algorithm implemented here can accommodate customisations in the maximum decision tree depth, the minimum sample size, the number of random features if the users want to choose randomly some `d` features without replacement when splitting a node, and the number of random splits if the users want to split a node for some `s` times and choose the best split among these `s` splits instead of choosing the best split among all potential splits of the node.

The algorithm is not able to work with datasets containing categorical data natively, so it requires those datasets to be preprocessed such as converting ordinal data into integers. After converting all categorical string values in a dataset to integers, a user can use that dataset with the algorithm.
## 2. Repository Structure
```
decision-tree-from-scratch/
├── dataset_files/
│   ├── breast_cancer.csv   # UCI Breast Cancer Wisconsin Diagnostic Dataset
│   └── car_evaluation.csv  # UCI Car Evaluation Dataset
│
├── decisionTree.py         # Decision Tree Algorithm
├── breastCancer.py         # Training and Testing on Breast Cancer Wisconsin (Diagnostic) Dataset
└── carEvaluation.py        # Training and Testing on Car Evaluation Dataset
```
## 3. Testing Specifications
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
## 4. Results on UCI Breast Cancer Wisconsin (Diagnostic) Dataset
Overfitting can be observed when the depth of the decision tree changes from 4 to 5, in which the testing accuracy decreases from 92.96% to 92.25% while the training accuracy increases from 98.13% to 99.30%.
```
maxDepth = 1: accTest = 88.73%, accTrain = 92.51%, buildTime = 2.99s
maxDepth = 2: accTest = 88.73%, accTrain = 93.68%, buildTime = 5.37s
maxDepth = 3: accTest = 92.96%, accTrain = 97.66%, buildTime = 6.68s
maxDepth = 4: accTest = 92.96%, accTrain = 98.13%, buildTime = 6.85s
maxDepth = 5: accTest = 92.25%, accTrain = 99.30%, buildTime = 6.94s
maxDepth = 6: accTest = 92.96%, accTrain = 99.77%, buildTime = 6.89s
maxDepth = 7: accTest = 92.96%, accTrain = 100.00%, buildTime = 6.98s
```
## 5. Results on UCI Car Evaluation Dataset
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
## 6. References
1. [Sebastian Mantey's repository](https://github.com/SebastianMantey/Decision-Tree-from-Scratch)
2. [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
3. [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
