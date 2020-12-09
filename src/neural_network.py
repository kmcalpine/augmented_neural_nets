from dataclasses import dataclass
from dataclasses import field
import collections

@dataclass
class Innovations:
    number: int = 0
    found: {} = field(default_factory=dict)
    
#

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
class Neuron:
    '''Class to represent the structure of a neuron.'''
    value: float
    connections: []
    neuron_type: NeuronType
    neuron_index: int

@dataclass
class Connection:
    '''Class for a Neuron's connections.'''
    from_n: Neuron
    to_n: Neuron
    weight: float 
    innovation: int

@dataclass
class NeuralNetwork:
    input_neurons: int
    output_neurons: int
    network_connections: [Connection]
    network_neurons: [Neuron]
    innovation: Innovations = Innovations
    node_index: int = 0

    def construct(self):

        def set_neurons():
            for i in range(self.input_neurons+self.output_neurons):
                if i < self.input_neurons:
                    
                    n_type = NeuronType(1,0,0)
                    n = Neuron(0, [], n_type, self.node_index)

                else:
                    n_type = NeuronType(0,0,1)
                    n = Neuron(0, [], n_type, self.node_index)

                self.node_index += 1
                self.network_neurons.append(n)

        
        def set_connections():
            for i in range(self.input_neurons):
                for j in range(self.input_neurons, self.input_neurons+self.output_neurons):
                        
                    conn = Connection(
                                            self.network_neurons[i],
                                            self.network_neurons[j],
                                            0.5,
                                            self.innovation.number
                                            )

                    self.network_connections.append(conn)

        set_neurons()
        set_connections()

neural_network = NeuralNetwork(5, 5, [], [])
neural_network.construct()

for neurons in neural_network.network_neurons:
    print(neurons)

for conns in neural_network.network_connections:
    print(conns)