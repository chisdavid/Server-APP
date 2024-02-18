# how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
import numpy as np
from sklearn.preprocessing import StandardScaler

from backend.AI.Models.Helpers.Model import Model
from keras.models import Sequential
from keras.layers import LSTM, Dropout
from keras.layers import Dense


class LSTM(Model):
    def __init__(self, dataset):
        scaler = StandardScaler()
        scaler = scaler.fit(dataset)
        dataset = scaler.transform(dataset)
        self.no_days_future = 1
        self.no_days_past = 14
        self.dataset = dataset
        super().__init__(dataset)

    def computeValues(self):
        x_train = []
        y_train = []
        for i in range(self.no_days_past, len(self.dataset) - self.no_days_future + 1):
            x_train.append(self.dataset[i - self.no_days_past:i, 0:self.dataset.shape[1]])
            y_train.append(self.dataset[i + self.no_days_future - 1:i + self.no_days_future, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        return x_train, y_train

        pass

    def trainNet(self):
        self.computeValues()
        model = Sequential()

        x_train, y_train = self.computeValues()
        model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0, 2))
        model.add(Dense(self.y_train.shape[1]))

        model.compile(optimizer='adam', loss='mse')
        model.summary()
        pass

    def displayPerformance(self):
        pass
