from abc import abstractmethod

class Model:
    def __init__(self, dataset):
        self.x_train = dataset[0]
        self.y_train = dataset[1]
        self.x_test = dataset[2]
        self.y_test = dataset[3]
        self.model = None

    @abstractmethod
    def trainNet(self):
        pass

    @abstractmethod
    def displayPerformance(self):
        pass

    @abstractmethod
    def getLoss(self):
        pass
