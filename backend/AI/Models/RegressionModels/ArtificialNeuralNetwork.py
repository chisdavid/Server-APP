import math
import os.path
from errno import ENOENT
from math import exp
from tkinter import *
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from Licenta.backend.AI.Models.Helpers.Layer import Layer
from Licenta.backend.AI.Models.Helpers.Model import Model
from Licenta.backend.AI.Models.Helpers.Neuron import Neuron
from Licenta.backend.AI.utils.enums import ActivationFunction
from pathlib import Path


class ArtificialNeuralNetwork(Model):

    def __init__(self, dataset, activationType, learningRate, noEpochs, noHiddenLayers, noNeuronsPerLayer,
                 randomWeights, load):
        super().__init__(dataset)
        self.net = None
        self.activationType = activationType
        self.learningRate = learningRate
        self.noEpochs = noEpochs
        self.noHiddenLayers = noHiddenLayers
        self.noNeuronsPerLayer = noNeuronsPerLayer
        self.randomWeights = randomWeights
        self.load = load
        if load:
            self.net = self.loadNet()
        else:
            self.net = self.buildNet()

    def buildNet(self):
        noInputs = len(self.x_train[1])
        noOutputs = 1

        firstLayer = Layer(noOutputs=self.noNeuronsPerLayer, noInputs=noInputs, randomWeights=self.randomWeights)
        net = [firstLayer]
        for i in range(self.noHiddenLayers):
            layer = Layer(noOutputs=self.noNeuronsPerLayer, noInputs=net[-1].noNeurons + 1,
                          randomWeights=self.randomWeights)
            net.append(layer)

        outputLayer = Layer(noOutputs=noOutputs, noInputs=net[-1].noNeurons, randomWeights=self.randomWeights)
        net.append(outputLayer)
        return net

    def loadNet(self):
        root = Tk()
        root.fileName = filedialog.askopenfilename(filetypes=(("howCode files", ".*"), ("All files", "*.*")))

        net = []
        neurons = []
        with open(root.fileName) as f:
            while True:
                line = f.readline()
                if line == '':
                    break

                if 'Layer' in line:
                    if neurons:
                        net.append(Layer(load=self.load, neurons=neurons))
                        neurons = []
                elif 'Weights' in line:
                    neuron = Neuron.fromString(self, line)
                    neurons.append(neuron)

        net.append(Layer(load=self.load, neurons=neurons))
        return net

    def getInput(self, inputs, weights):
        result = weights[-1]
        for i in range(len(inputs)):
            result += inputs[i] * weights[i]
        return result

    def forwardPropagation(self, inputs):
        for layer in self.net:
            newInputs = []
            for neuron in layer.get():
                activation = self.getInput(inputs, neuron.weights)
                neuron.output = self.activate(activation)
                newInputs.append(neuron.output)
            inputs = newInputs
        return inputs

    def activate(self, value):
        if self.activationType == ActivationFunction.SIGMOID:
            return 1.0 / (1.0 + exp(-value))

        if self.activationType == ActivationFunction.SOFTPLUS:
            return np.log(1 + exp(value))

        if self.activationType == ActivationFunction.THAN:
            return -1 + 2.0 / (1 + exp(-2 * value))

        if self.activationType == ActivationFunction.ATAN:
            return 1 / math.atan(value)

    def errorDerivate(self, value):
        if self.activationType == ActivationFunction.SIGMOID:
            return value * (1 - value)

        if self.activationType == ActivationFunction.SOFTPLUS:
            return (exp(value) - 1) / (exp(value))

        if self.activationType == ActivationFunction.THAN:
            return 1 - value ** 2

        if self.activationType == ActivationFunction.ATAN:
            return 1.0 / (1 + value * value)

    def backwardPropagationError(self, expected):
        for i in reversed(range(len(self.net))):
            currentLayer = self.net[i].get()
            errors = list()
            if i != len(self.net) - 1:  # hidden layers
                for j in range(len(currentLayer)):
                    error = 0.0
                    for neuron in self.net[i + 1].get():
                        error += neuron.weights[j] * neuron.delta
                    errors.append(error)

            else:  # output layer
                for j in range(len(currentLayer)):
                    neuron = currentLayer[j]
                    errors.append(neuron.output - expected[j])

            for j in range(len(currentLayer)):
                neuron = currentLayer[j]
                neuron.delta = errors[j] * self.errorDerivate(neuron.output)

    def updateWeights(self, inputs):
        for i in range(len(self.net)):
            currentLayer = self.net[i].get()
            newInputs = inputs
            if i != 0:
                newInputs = [neuron.output for neuron in self.net[i - 1].get()]

            for neuron in currentLayer:
                for j in range(len(newInputs)):
                    neuron.weights[j] -= self.learningRate * neuron.delta * newInputs[j]
                neuron.weights[-1] -= self.learningRate * neuron.delta

    def Save(self, filePath, loss):
        try:
            with open(filePath, 'w') as f:
                for index, value in enumerate(self.net):
                    layer = value.getLayerAsString()
                    if layer is not None:
                        f.write(f'Layer {index}')
                        f.write(f'{layer}\n')
                f.write(f'\nLOSS = {loss}')
        except IOError as error:
            print(error)

    def saveModel(self):
        loss = self.getLoss()
        pathName = os.path.join(os.getcwd(), 'Results', f'{self.activationType.value}',
                                f'Model_{self.noHiddenLayers}_{self.noNeuronsPerLayer}.txt')
        try:
            f = open(pathName)
            lastLOSS = float(f.readlines()[-1].split(' ')[-1])
            print(f'loss  {loss} {lastLOSS}')
            if loss < lastLOSS:
                self.Save(pathName, loss)

        except IOError as error:
            if error.errno == ENOENT:
                self.Save(pathName, loss)

    def trainNet(self):
        errors = [999999999999999999999]
        bestNoEpchs = 0
        for epoch in range(self.noEpochs):
            outputs = []
            sumError = 0.0
            for inputs, expected in zip(self.x_train, self.y_train):
                computedOutputs = self.forwardPropagation(inputs)
                outputs.append(computedOutputs[0])
                currentError = sum([1/len(expected) * abs(expected[i] - computedOutputs[i]) ** 2 for i in range(0, len(expected))])
                sumError += currentError
                self.backwardPropagationError(expected)
                self.updateWeights(inputs)

                # print(self.net)

            array = np.linspace(0.1, 0.8, 100)
            plt.plot(self.y_train, outputs, color='red', label='Real data / Expected Data')
            requests.post('http://')
            plt.plot(array, array, color='green', label='Perfect Data')
            plt.title("Date Reale / Date Prezise")
            plt.legend()
            plt.show()
            print(self.getLoss())

            bestNoEpchs = epoch
            if (sumError > errors[-1] or errors[-1] - sumError < 0.5):
                break
            else:
                errors.append(sumError)

            print('> Epoch=%d/%d, lrate=%.6f, error=%.6f' % (epoch, self.noEpochs, self.learningRate, sumError),
                  end='\n\n')

        self.saveModel()

        return bestNoEpchs + 1

    def predict(self, data):
        computedOutputs = []
        for inputs in data:
            outputs = self.forwardPropagation(inputs)
            computedOutputs.append(outputs)
        return computedOutputs

    def getLoss(self):
        data = self.predict(self.x_test)
        predictedData = [i[0] for i in data]
        actualsData = [i[0] for i in self.y_test]
        return sum([abs(predictedData[i] - actualsData[i]) for i in range(len(actualsData))])

    def displayNet(self):
        for index, layer in enumerate(self.net):
            print(f'Layer {index}')
            layer.toString()

    def displayPerformance(self):
        y_pred = self.predict(self.x_test)
        # for i in range(len(y_pred)):
        #     print('>Actuals = %.10f, Expected=%.10f' % (y_pred[i][0], self.y_test[i][0]))

        df1 = pd.DataFrame(y_pred)
        df2 = pd.DataFrame(self.y_test)
        plt.plot(df2[0], color='red', label='Real data')
        plt.plot(df1[0], color='blue', label='Predicted data')

        plt.title('Prediction')
        plt.legend()
        plt.show()
        pass
