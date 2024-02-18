from collections import Counter


class Neuron:
    def __init__(self, weights=[], output=0, delta=0.0):
        self.weights = weights
        self.output = output
        self.delta = delta

    def toString(self):
        print(self.getNeuronAsString())

    def getNeuronAsString(self):
        return f'\tWeights: {self.weights}; Output: {self.output}; Delta: {self.delta}'

    def fromString(self, line):
        parts = line.split(';')

        # Weights
        parts[0] = parts[0].translate({ord(i): None for i in ' []:;Weights'})
        weights = [float(i) for i in parts[0].split(',')]

        # Output
        output = float(parts[1].translate({ord(i): None for i in 'Output :'}))

        # Delta
        parts[2] = parts[2].translate({ord(i): None for i in 'Delta: \n'})
        if '-' in parts[2]:
            if parts[2][0] == '-':
                counter = Counter(parts[2])
                if counter['-'] == 2:
                    data = parts[2].split('-')
                    number = float(data[1])
                    delta = -number / int(data[2])
                else:
                    delta = float(parts[2])
            else:
                data = parts[2].split('-')
                number = float(data[0])
                delta = number / int(data[1])

        else:
            delta = float(parts[2])
        return Neuron(weights=weights, output=output, delta=delta)
