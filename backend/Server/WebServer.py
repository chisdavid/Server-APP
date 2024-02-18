import json

from flask import Flask, request
from Licenta.backend.Repository.Repository import Repository
from Licenta.backend.Routes.Routes import main
from flask_cors import CORS, cross_origin
from flask import jsonify

from Licenta.backend.Server.ServiceImplementation import getTrainedModels

from Licenta.backend.AI.utils.enums import DatasetName

app = Flask(__name__)
CORS(app)

repository = Repository(DatasetName.DATA_ALL_OBSERVATION)


@app.route(main)
def hello_word():
    # repository.artificialNeuralNetworkModel()
    return "<p>Hello World!</p>"


@app.route("/Ana")
def hey():
    return 'Ana'


@app.route("/ReadTrainedRoutes", methods=['GET'], strict_slashes=False)
@cross_origin()
def getTrainedNets():
    return jsonify(getTrainedModels())


@app.route("/TrainNet", methods=['POST'])
@cross_origin()
def trainNet():
    noEpochs = int(request.json['noEpochs'])
    noNeuronsPerLayer = int(request.json['noNeuronsPerLayer'])
    noHiddenLayers = int(request.json['noHiddenLayers'])
    learningRate = request.json['learningRate']
    activationFunction = str(request.json['activationFunction'])
    randomWeights = bool(request.json['randomWeights'])

    repository.artificialNeuralNetworkModel1(noHiddenLayers,noEpochs,learningRate,noNeuronsPerLayer,randomWeights,activationFunction)



    return "End"


if __name__ == '__main__':
    # print(2+3)
    app.run(host='0.0.0.0', port='5000', debug=False)
