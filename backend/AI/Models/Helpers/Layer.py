from random import random
import numpy as np
from Licenta.backend.AI.Models.Helpers.Neuron import Neuron
from random import gauss


class Layer:
    def __init__(self, noOutputs=0, noInputs=0, randomWeights=True, load=False, neurons=[]):
        self.layer = None
        self.noInputs = noInputs
        self.noNeurons = noOutputs
        self.randomWeights = randomWeights

        if load:
            self.layer = neurons
        else:
            self.buildLayer()

    def buildLayer(self):
        if self.randomWeights:
            self.layer = [Neuron(weights=[gauss(0, 1) for j in range(self.noInputs + 1)], delta=0, output=0) for i in
                          range(self.noNeurons)]
        else:
            self.layer = [Neuron(weights=[0 for j in range(self.noInputs + 1)], delta=0, output=0) for i in
                          range(self.noNeurons)]

    def get(self) -> list:
        return self.layer

    def toString(self):
        for index, neuron in enumerate(self.layer):
            print(f'Neuron {index}')
            print(neuron.toString())

    def getLayerAsString(self) -> str:
        layer = ''
        for index, neuron in enumerate(self.layer):
            layer = f'{layer}\n {neuron.getNeuronAsString()}'

        return layer
