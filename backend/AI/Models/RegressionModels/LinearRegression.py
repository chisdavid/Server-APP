from abc import ABC

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as Regression

from Licenta.backend.AI.Models.Helpers.Model import Model


class LinearRegression(Model, ABC):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.net = Regression()

    def trainNet(self):
        self.net.fit(self.x_train, self.y_train)

    def displayPerformanceRegression(self):
        y_pred = self.net.predict(self.x_test)
        y_test = self.y_test
        for i in range(0, len(y_pred)):
            print('> Actuals ' + str(y_pred[i][0]) + ' / Expected =' + str(y_test[i][0]))

        print(self.net.coef_)
        plt.plot(y_pred)
        plt.plot(y_test)
        plt.show()
        df1 = pd.DataFrame(y_pred)
        df2 = pd.DataFrame(y_test)
        #
        fix, ax = plt.subplots()
        plt.plot(df1[0], color='red', label='Real data')
        plt.plot(df2[0], color='blue', label='Predicted data')
        plt.title('Prediction')
        plt.legend()
        plt.show()
