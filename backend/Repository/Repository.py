from Licenta.backend.AI.Models.RegressionModels.Ann import Ann
from Licenta.backend.AI.Models.RegressionModels.ArtificialNeuralNetwork import ArtificialNeuralNetwork
from Licenta.backend.AI.utils.enums import ActivationFunction
from Licenta.backend.AI.utils.methods import splitDataFrame, convertDataset
from Licenta.backend.AI.Models.RegressionModels.LinearRegression import LinearRegression


class Repository:
    def __init__(self, dataset):
        self.dataset = splitDataFrame(dataset)

    def linearRegression(self):
        dataset = convertDataset(self.dataset)
        linearRegression = LinearRegression(dataset)
        linearRegression.trainNet()
        linearRegression.displayPerformanceRegression()

    def artificialNeuralNetworkModel1(self, noHiddenLayers=None, noEpochs=None, learningRate=None,
                                      noNeuronsPerLayer=None, randomWeights=None,activationType=None):
        dataset = convertDataset(self.dataset)
        ann = ArtificialNeuralNetwork(
            dataset=dataset,
            activationType=ActivationFunction['SIGMOID'],
            learningRate=learningRate,
            noEpochs=noEpochs,
            noHiddenLayers=noHiddenLayers,
            noNeuronsPerLayer=noNeuronsPerLayer,
            randomWeights=randomWeights,
            load=False
        )

        ann.trainNet()

        # self.linearRegression()
        ann.displayPerformance()

        pass

    def artificialNeuralNetworkModel(self):

        # 0.0009 100 epochs SoftMax 8 si 1
        noEpochs = 100  # 83 #50
        learningRate = 0.001  # 0.001
        noNeuronsPerLayer = 16  # 8 #16
        noHiddenLayers = 2  # 1 #2
        # error 0 #100
        activationType = ActivationFunction.SIGMOID
        dataset = convertDataset(self.dataset)
        randomWeights = True
        load = False

        ann = ArtificialNeuralNetwork(
            dataset=dataset,
            activationType=activationType,
            learningRate=learningRate,
            noEpochs=noEpochs,
            noHiddenLayers=noHiddenLayers,
            noNeuronsPerLayer=noNeuronsPerLayer,
            randomWeights=randomWeights,
            load=load
        )

        if not load:
            ann.trainNet()

        # self.linearRegression()
        ann.displayPerformance()

    def ann(self):
        noEpochs = 50
        learningRate = 0.001
        noNeuronsPerLayer = 8
        noHiddenLayers = 1
        activationType = ActivationFunction.SIGMOID
        dataset = convertDataset(self.dataset)
        randomWeights = True

        ann = Ann(
            dataset=dataset,
            activationType=activationType,
            learningRate=learningRate,
            noEpochs=noEpochs,
            noHiddenLayers=noHiddenLayers,
            noNeuronsPerLayer=noNeuronsPerLayer,
            randomWeights=randomWeights
        )

        ann.trainNet()
        ann.displayPerformance()
        print("Final")
