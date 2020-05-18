import random
import time
from decisionTree import loadData, buildDecisionTree, decisionTreePredictions, calculateAccuracy

xTrainFile = "dataset_files/car_X_train.csv"
xTestFile = "dataset_files/car_X_test.csv"
yTrainFile = "dataset_files/car_y_train.csv"
yTestFile = "dataset_files/car_y_test.csv"
dataFrameTrain, dataFrameTest = loadData(xTrainFile, xTestFile, yTrainFile, yTestFile)

buyingMapping = {"low": 1, "med": 2, "high": 3, "vhigh": 4}
dataFrameTrain["buying"] = dataFrameTrain["buying"].map(buyingMapping)
dataFrameTest["buying"] = dataFrameTest["buying"].map(buyingMapping)

maintMapping = {"low": 1, "med": 2, "high": 3, "vhigh": 4}
dataFrameTrain["maint"] = dataFrameTrain["maint"].map(maintMapping)
dataFrameTest["maint"] = dataFrameTest["maint"].map(maintMapping)

doorsMapping = {"2": 2, "3": 3, "4": 4, "5more": 5}
dataFrameTrain["doors"] = dataFrameTrain["doors"].map(doorsMapping)
dataFrameTest["doors"] = dataFrameTest["doors"].map(doorsMapping)

personsMapping = {"2": 2, "4": 4, "more": 6}
dataFrameTrain["persons"] = dataFrameTrain["persons"].map(personsMapping)
dataFrameTest["persons"] = dataFrameTest["persons"].map(personsMapping)

lugBootMapping = {"small": 1, "med": 2, "big": 3}
dataFrameTrain["lug_boot"] = dataFrameTrain["lug_boot"].map(lugBootMapping)
dataFrameTest["lug_boot"] = dataFrameTest["lug_boot"].map(lugBootMapping)

safetyMapping = {"low": 1, "med": 2, "high": 3}
dataFrameTrain["safety"] = dataFrameTrain["safety"].map(safetyMapping)
dataFrameTest["safety"] = dataFrameTest["safety"].map(safetyMapping)

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
