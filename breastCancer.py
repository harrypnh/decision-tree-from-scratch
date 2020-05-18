import random
import time
from decisionTree import loadData, buildDecisionTree, decisionTreePredictions, calculateAccuracy

xTrainFile = "dataset_files/cancer_X_train.csv"
xTestFile = "dataset_files/cancer_X_test.csv"
yTrainFile = "dataset_files/cancer_y_train.csv"
yTestFile = "dataset_files/cancer_y_test.csv"
dataFrameTrain, dataFrameTest = loadData(xTrainFile, xTestFile, yTrainFile, yTestFile)

print("Decision Tree - Breast Cancer Dataset")

i = 1
accuracyTrain = 0
while accuracyTrain < 100:
    startTime = time.time()
    decisionTree = buildDecisionTree(dataFrameTrain, maxDepth = i)
    buildingTime = time.time() - startTime
    decisionTreeTestResults = decisionTreePredictions(dataFrameTest, decisionTree)
    accuracyTest = calculateAccuracy(decisionTreeTestResults, dataFrameTest.iloc[:, -1]) * 100
    decisionTreeTrainResults = decisionTreePredictions(dataFrameTrain, decisionTree)
    accuracyTrain = calculateAccuracy(decisionTreeTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("maxDepth = {}: ".format(i), end = "")
    print("accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")
    i += 1
