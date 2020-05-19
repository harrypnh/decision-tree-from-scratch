import random
import pandas
import time
from decisionTree import trainTestSplit, buildDecisionTree, decisionTreePredictions, calculateAccuracy

dataFrame = pandas.read_csv("dataset_files/car_evaluation.csv")

buyingMapping = {"low": 1, "med": 2, "high": 3, "vhigh": 4}
dataFrame["buying"] = dataFrame["buying"].map(buyingMapping)

maintMapping = {"low": 1, "med": 2, "high": 3, "vhigh": 4}
dataFrame["maint"] = dataFrame["maint"].map(maintMapping)

doorsMapping = {"2": 2, "3": 3, "4": 4, "5more": 5}
dataFrame["doors"] = dataFrame["doors"].map(doorsMapping)

personsMapping = {"2": 2, "4": 4, "more": 6}
dataFrame["persons"] = dataFrame["persons"].map(personsMapping)

lugBootMapping = {"small": 1, "med": 2, "big": 3}
dataFrame["lug_boot"] = dataFrame["lug_boot"].map(lugBootMapping)

safetyMapping = {"low": 1, "med": 2, "high": 3}
dataFrame["safety"] = dataFrame["safety"].map(safetyMapping)

dataFrameTrain, dataFrameTest = trainTestSplit(dataFrame, testSize = 0.3)

print("Decision Tree - Car Evaluation Dataset")

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
