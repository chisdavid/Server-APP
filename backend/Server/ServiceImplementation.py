import os


def getTrainedModels():
    resultsDir = os.path.abspath("../AI/Results")
    models = []
    for folder in os.listdir(resultsDir):
        path = f'{resultsDir}\{folder}'
        for file in os.listdir(path):
            if 'Model' in file:
                name = f'{file}_{folder}'
                name = name.replace('.txt', '')
                words = name.split('_')
                modelName = f'{words[1]} Layers,{words[2]} Neurons/Layer, ActivationType {words[3]}'
                print(modelName)
                models.append(modelName)
    print(models)
    return models
