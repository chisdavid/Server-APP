from enum import Enum


class ActivationFunction(Enum):
    SIGMOID = 'Sigmoid'
    SOFTPLUS = 'Softplus'
    THAN = 'Than'
    ATAN = 'Atan'


class DatasetName(Enum):
    DATA_ALL_OBSERVATION = 'data_all_observations.csv'
    HEAT_PUMP = 'heatpumpDataset.csv'
