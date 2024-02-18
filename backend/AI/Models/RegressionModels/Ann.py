import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from Licenta.backend.AI.Models.Helpers.Model import Model


class Ann(Model):
    def __init__(self, dataset, activationType, learningRate, noEpochs, noHiddenLayers, noNeuronsPerLayer,
                 randomWeights):
        super(Ann, self).__init__(dataset)
        self.net = None
        self.activationType = activationType
        self.learningRate = learningRate
        self.noEpochs = noEpochs
        self.noHiddenLayers = noHiddenLayers
        self.noNeuronsPerLayer = noNeuronsPerLayer
        self.randomWeights = randomWeights
        self.buildNet()

    #
    def buildNet(self):
        noInputs = len(self.x_train[1])
        model = Sequential()
        model.add(Dense(self.noNeuronsPerLayer, input_shape=[noInputs, ], activation='sigmoid'))

        for i in range(self.noHiddenLayers):
            model.add(Dense(self.noNeuronsPerLayer, activation=self.activationType.value))

        model.add(Dense(1))
        self.net = model

    #
    def trainNet(self):
        self.net.compile(loss='mse', optimizer='adam', metrics=['accuracy'], )
        self.net.fit(
            self.x_train,
            self.y_train,
            verbose=1,
            epochs=self.noEpochs,
            batch_size=32,
            validation_data=(
                self.x_test,
                self.y_test
            ))

    def displayPerformance(self):
        y_pred = self.net.predict(self.x_test)
        #
        # for i in range(len(y_pred)):
        #     print('>Actuals = %.10f, Expected=%.10f' % (y_pred[i], list(self.y_test[getPredictedVariableName(DatasetName.HEAT_PUMP)])[i]))

        df1 = pd.DataFrame(y_pred)
        df2 = pd.DataFrame(self.y_test)

        fix, ax = plt.subplots()
        plt.plot(df2[0], color='red', label='Real data')
        plt.plot(df1[0], color='blue', label='Predicted data')
        plt.title('Prediction')
        plt.legend()
        plt.show()
