import csv
import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from Licenta.backend.AI.Models.RegressionModels.ArtificialNeuralNetwork import ArtificialNeuralNetwork
from Licenta.backend.AI.utils.enums import DatasetName, ActivationFunction


def getRow(inputs, selectedIndexes):
    row = []
    for i in selectedIndexes:
        row.append(float(inputs[i]))

    if 0 in row:
        return []
    return row


def splitDataFrame(datasetName):
    output = getPredictedVariableName(datasetName)
    dataset = readData(datasetName)
    x = dataset.loc[:, dataset.columns != output]
    y = dataset.loc[:, dataset.columns == output]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0,shuffle=False)
    return x_train, y_train, x_test, y_test


def convertDataset(data):
    return [dataFrameToList(i) for i in data]


def getDataset():
    return convertDataset(splitDataFrame())


def getFilePath(datasetName):
    filePath = os.path.dirname(__file__)
    return os.path.join(filePath.replace('\\utils',''),'Datasets',datasetName.value)


def getSelectedColumns(datasetName):
    if (datasetName == DatasetName.DATA_ALL_OBSERVATION):
        return ['hm_brine_flow', 'hm_brine_returntemp_c', 'hm_brine_supplytemp_c', 'hm_heat_flow','hm_heat_thermal_power_w', 'pm_active_power_w','pm_apparent_power_w']

    if datasetName == DatasetName.HEAT_PUMP:
        return ['Room Air Temperature', 'Ambient Outdoor Temperature', 'Compressor Power', 'Fan Power',
                'Total Unit Power']


def getPredictedVariableName(datasetName):
    if datasetName == DatasetName.DATA_ALL_OBSERVATION:
        return 'hm_heat_thermal_power_w'

    if datasetName == DatasetName.HEAT_PUMP:
        return 'Total Unit Power'


def readData(datasetName):
    data = []
    dataNames = []

    filePath = getFilePath(datasetName)
    selectedColumns = getSelectedColumns(datasetName)
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        selectedIndexes = []
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                if not selectedIndexes:
                    selectedIndexes = [dataNames.index(i) for i in selectedColumns]
                # dataRow = [float(row[i]) for i in selectedIndexes]
                dataRow = getRow(row, selectedIndexes)
                if dataRow:
                    data.append(dataRow)
            line_count += 1

        data = rescaleDataset(data)

        dataset = pd.DataFrame(data)
        dataset.columns = [dataNames[i] for i in selectedIndexes]
        return rescaleDataSet(selectedColumns, dataset)


def rescaleDataset(data):
    for i in range(len(data[0])):
        maxim = -1111
        for j in range(len(data)):
            maxim = max(maxim, data[j][i])

        for j in range(len(data)):
            data[j][i] /= maxim

    return data


def rescaleDataSet(selectedColumns, dataset):
    scaler = MaxAbsScaler()
    # MaxAbsScaler
    datasetScaled = scaler.fit_transform(dataset)
    datasetScaled = pd.DataFrame(datasetScaled, columns=selectedColumns)

    return datasetScaled


def dataFrameToList(dataframe):
    outputList = []
    for i in dataframe:
        innerList = []
        for j in dataframe[i]:
            innerList.append(j)
        outputList.append(innerList)

    return np.array(outputList).transpose().astype('float32')


def findOptimParameters():
    learningRateArray = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    noEpochsArray = [50, 100, 150, 200]
    noNeuronsPerLayerArray = [4, 8, 16]
    noHiddenLayersArray = [0, 1, 2, 3]
    activationType = ActivationFunction.SIGMOID
    randomWeights = True
    dataset = getDataset()
    for noNeuronsPerLayer in noNeuronsPerLayerArray:
        for noHiddenLayers in noHiddenLayersArray:
            for learningRate in learningRateArray:
                noPreviousEpochs = 0
                bestEpochNumber = 1000
                bestLoss = 9999999999
                loss = 0
                for noEpochs in noEpochsArray:

                    if bestEpochNumber >= noPreviousEpochs and loss < bestLoss:
                        noPreviousEpochs = noEpochs
                        ann = ArtificialNeuralNetwork(
                            dataset=dataset,
                            activationType=activationType,
                            learningRate=learningRate,
                            noEpochs=noEpochs,
                            noHiddenLayers=noHiddenLayers,
                            noNeuronsPerLayer=noNeuronsPerLayer,
                            randomWeights=randomWeights
                        )
                        bestEpochNumber = ann.trainNet()
                        loss = ann.getLoss()
                        bestLoss = loss

                    open("data.txt", "a").write(
                        '> noEpochs=%d, bestEpochNumber=%d learningRate=%.6f, noNeuronsPerLayer=%d noHiddenLayers=%d \nLOSS=%.6f \n\n' % (
                            noEpochs, bestEpochNumber, learningRate, noNeuronsPerLayer, noHiddenLayers, loss))

                    # print('> noEpochs=%d, lrate=%.6f, noNeuronsPerLayer=%d noHiddenLayers=%d' % (
                    #     noEpochs, learningRate, noNeuronsPerLayer, noHiddenLayers))
                    # print("loss = " + str(ann.getLoss()))
    print("Finally ")
