from dataclasses import dataclass

@dataclass
class NeuronType:
    '''Class to represent the Neuron type.'''
    input: int = 0
    hidden: int = 0
    output: int = 0

    def __repr__(self):
        if self.input == 1:
            return 'input'
        if self.hidden == 1:
            return 'hidden'
        if self.output == 1:
            return 'output'

@dataclass
class Connection:
    '''Class for a Neuron's connections.'''
    incoming: [int]
    outgoing: [int]

@dataclass
class Neuron:
    '''Class to represent the structure of a neuron.'''
    weight: float
    connections: Connection
    neuron_type: NeuronType
    neuron_index: int

@dataclass
class NeuralNetwork:
    input_neurons: int
    output_neurons: int
    network_connections: [Connection]
    network_neurons: [Neuron]

n_weight = 0.1
n_connection = Connection([], [])
n_type = NeuronType(1,0,0)
n = Neuron(n_weight, n_connection, n_type, 1)

n_weight = 0.25
n_connection = Connection([], [])
n_type = NeuronType(0,0,1)
n2 = Neuron(n_weight, n_connection, n_type, 2)

n.connections.outgoing.append(2)
n2.connections.incoming.append(1)

neural_network = NeuralNetwork(0, 0, [], [])
print(n)
print(n2)
print(n.__doc__)