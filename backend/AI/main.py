from Repository.Repository import Repository
from AI.utils.enums import DatasetName

repository = Repository(DatasetName.DATA_ALL_OBSERVATION)

if __name__ == '__main__':
    repository.artificialNeuralNetworkModel()
    # repository.ann()
    # repository.linearRegression()