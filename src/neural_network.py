from dataclasses import dataclass
from dataclasses import field
from typing import Dict
import collections

@dataclass
class Innovations:
    '''Class to represent the innovation number of 
       a given connection.'''
    number: int
    found: Dict = field(default_factory=dict)

@dataclass
class NeuronType:
    '''Class to represent the Neuron type.'''
    input: int = 0
    hidden: int = 0
    output: int = 0

    #debugging help
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
    in_connections: []
    out_connections: []
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
    hidden_neurons: int
    output_neurons: int
    network_connections: [Connection]
    network_neurons: [Neuron]
    innovation: Innovations = Innovations(0, {})
    node_index: int = 0

    def network_size(self):
        return self.input_neurons + self.hidden_neurons + self.output_neurons


    def construct(self):
        '''Methods to construct a neural network.'''
        def set_neurons():
            for i in range(self.network_size()):
                if i < self.input_neurons: # create input neuron
                    n_type = NeuronType(1,0,0)
                    n = Neuron(1, [], [], n_type, self.node_index)
                elif i < self.input_neurons+self.hidden_neurons and self.hidden_neurons: # create hidden neuron
                    n_type = NeuronType(0,1,0)
                    n = Neuron(0, [], [], n_type, self.node_index)
                else: # create output neuron
                    n_type = NeuronType(0,0,1)
                    n = Neuron(0, [], [], n_type, self.node_index)

                self.node_index += 1
                self.network_neurons.append(n)

        def set_connections():
            # set input to hidden neuron connections
            for i in range(self.input_neurons):
                for j in range(self.input_neurons, self.input_neurons+self.output_neurons):
                        
                    conn = Connection(
                                        self.network_neurons[i], # from neuron
                                        self.network_neurons[j], # to neuron
                                        1, # connection weight
                                        self.innovation.number # innovation number
                                        )

                    self.innovation.number += 1
                    self.innovation.found[str(i)+'->'+str(j)] = conn.innovation
                    self.network_connections.append(conn)
                    self.network_neurons[i].out_connections.append(conn)
                    self.network_neurons[j].in_connections.append(conn)
        
        def set_fixed_connections():
            # set input to hidden neuron connections
            for i in range(self.input_neurons):
                for j in range(self.input_neurons, self.input_neurons+self.hidden_neurons):
                        
                    conn = Connection(
                                        self.network_neurons[i], # from neuron
                                        self.network_neurons[j], # to neuron
                                        1, # connection weight
                                        self.innovation.number # innovation number
                                        )

                    self.innovation.number += 1
                    self.innovation.found[str(i)+'->'+str(j)] = conn.innovation
                    self.network_connections.append(conn)
                    self.network_neurons[i].out_connections.append(conn)
                    self.network_neurons[j].in_connections.append(conn)

            # set hidden to output neuron connections
            for i in range(self.input_neurons, self.input_neurons+self.hidden_neurons):
                for j in range(self.input_neurons+self.hidden_neurons, self.input_neurons+self.hidden_neurons+self.output_neurons):
                        
                    conn = Connection(
                                        self.network_neurons[i], # from neuron
                                        self.network_neurons[j], # to neuron
                                        1, # connection weight
                                        self.innovation.number # innovation number
                                        )

                    self.innovation.number += 1
                    self.innovation.found[str(i)+'->'+str(j)] = conn.innovation
                    self.network_connections.append(conn)
                    self.network_neurons[i].out_connections.append(conn)
                    self.network_neurons[j].in_connections.append(conn)

        set_neurons()
        set_connections()


    def predict(self, neuron_inputs):
        '''Method to predict neuron output values.'''
        for i in range(len(neuron_inputs)):
            if neuron_inputs[i].from_n.neuron_type != 'input':  
                self.predict(neuron_inputs[i].from_n.in_connections)
            neuron_inputs[i].to_n.value += neuron_inputs[i].from_n.value * neuron_inputs[i].weight

            if i == len(neuron_inputs)-1:
                # apply ReLU
                neuron_inputs[i].to_n.value = max(0, neuron_inputs[i].to_n.value)

neural_network = NeuralNetwork(5, 0, 5, [], [])
neural_network.construct()

for i in range(neural_network.input_neurons,
               neural_network.network_size()
               ):

    if neural_network.network_neurons[i].neuron_type.output:
        neural_network.predict(neural_network.network_neurons[i].in_connections)


for i in range(neural_network.input_neurons+neural_network.output_neurons):
    print(neural_network.network_neurons[i].value)

print(len(neural_network.network_connections))