import random
from randomForest import *
from decisionTree import *

xTrainFile = "dataset_files/cancer_X_train.csv"
xTestFile = "dataset_files/cancer_X_test.csv"
yTrainFile = "dataset_files/cancer_y_train.csv"
yTestFile = "dataset_files/cancer_y_test.csv"
dataFrameTrain, dataFrameTest = loadData(xTrainFile, xTestFile, yTrainFile, yTestFile)

print("Breast Cancer Datasets")
print("Method 1: Decision Tree")

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
    print("  maxDepth = {}: ".format(i))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")
    i += 1

print("\nMethod 2: Random Forest")
print("  Maximum bootstrap size (n) is {}".format(dataFrameTrain.shape[0]))
print("  Maximum random subspace size (d) is {}".format(dataFrameTrain.shape[1] - 1))

print("\n  Change n, keep other parameters")
for i in range(10, dataFrameTrain.shape[0] + 1, 50):
    randomForest, buildingTime = createRandomForest(dataFrameTrain, bootstrapSize = i,
                                                    randomAttributes = 10, randomSplits = 50,
                                                    forestSize = 30, treeMaxDepth = 3)
    randomForestTestResults = randomForestPredictions(dataFrameTest, randomForest)
    accuracyTest = calculateAccuracy(randomForestTestResults, dataFrameTest.iloc[:, -1]) * 100
    randomForestTrainResults = randomForestPredictions(dataFrameTrain, randomForest)
    accuracyTrain = calculateAccuracy(randomForestTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("  n = {}, d = {}, s = {}, k = {}, maxDepth = {}:".format(i, 10, 50, 30, 3))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")

print("\n  Change d, keep other parameters")
for i in range(10, dataFrameTrain.shape[1], 2):
    randomForest, buildingTime = createRandomForest(dataFrameTrain, bootstrapSize = 60,
                                                    randomAttributes = i, randomSplits = 50,
                                                    forestSize = 30, treeMaxDepth = 3)
    randomForestTestResults = randomForestPredictions(dataFrameTest, randomForest)
    accuracyTest = calculateAccuracy(randomForestTestResults, dataFrameTest.iloc[:, -1]) * 100
    randomForestTrainResults = randomForestPredictions(dataFrameTrain, randomForest)
    accuracyTrain = calculateAccuracy(randomForestTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("  n = {}, d = {}, s = {}, k = {}, maxDepth = {}:".format(60, i, 50, 30, 3))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")

print("\n  Change s, keep other parameters")
for i in range(10, 100 + 1, 10):
    randomForest, buildingTime = createRandomForest(dataFrameTrain, bootstrapSize = 60,
                                                    randomAttributes = 10, randomSplits = i,
                                                    forestSize = 30, treeMaxDepth = 3)
    randomForestTestResults = randomForestPredictions(dataFrameTest, randomForest)
    accuracyTest = calculateAccuracy(randomForestTestResults, dataFrameTest.iloc[:, -1]) * 100
    randomForestTrainResults = randomForestPredictions(dataFrameTrain, randomForest)
    accuracyTrain = calculateAccuracy(randomForestTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("  n = {}, d = {}, s = {}, k = {}, maxDepth = {}:".format(60, 10, i, 30, 3))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")

print("\n  Change k, keep other parameters")
for i in range(10, 100 + 1, 10):
    randomForest, buildingTime = createRandomForest(dataFrameTrain, bootstrapSize = 60,
                                                    randomAttributes = 10, randomSplits = 50,
                                                    forestSize = i, treeMaxDepth = 3)
    randomForestTestResults = randomForestPredictions(dataFrameTest, randomForest)
    accuracyTest = calculateAccuracy(randomForestTestResults, dataFrameTest.iloc[:, -1]) * 100
    randomForestTrainResults = randomForestPredictions(dataFrameTrain, randomForest)
    accuracyTrain = calculateAccuracy(randomForestTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("  n = {}, d = {}, s = {}, k = {}, maxDepth = {}:".format(60, 10, 50, i, 3))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")