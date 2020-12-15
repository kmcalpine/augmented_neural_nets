from dataclasses import dataclass
from dataclasses import field
from random import random
from typing import Dict
import collections
import math


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

        # find a better implementation to remove duplicate code
        def set_connections():
            # set input to hidden neuron connections
            for i in range(self.input_neurons):
                for j in range(self.input_neurons, self.input_neurons+self.output_neurons):
                        
                    conn = Connection(
                                        self.network_neurons[i], # from neuron
                                        self.network_neurons[j], # to neuron
                                        (random()*2)-1, # connection weight
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
                                        (random()*2)-1, # connection weight
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
                                        (random()*2)-1, # connection weight
                                        self.innovation.number # innovation number
                                        )

                    self.innovation.number += 1
                    self.innovation.found[str(i)+'->'+str(j)] = conn.innovation
                    self.network_connections.append(conn)
                    self.network_neurons[i].out_connections.append(conn)
                    self.network_neurons[j].in_connections.append(conn)

        set_neurons()
        set_fixed_connections()


    def predict(self, neuron_inputs):
        '''Method to predict neuron output values.'''
        for i in range(len(neuron_inputs)):
            if neuron_inputs[i].from_n.neuron_type != 'input':  
                self.predict(neuron_inputs[i].from_n.in_connections)
            neuron_inputs[i].to_n.value += neuron_inputs[i].from_n.value * neuron_inputs[i].weight

            if i == len(neuron_inputs)-1:
                # apply ReLU
                neuron_inputs[i].to_n.value = max(0, neuron_inputs[i].to_n.value)


    def mutate(self):
        val = 5
        def mutate_weight(x):
            # randomly select a network connection and mutate its weight
            conn = self.network_connections[math.floor(random()*len(self.network_connections))]
            conn.weight += ((random()*2)-1)
            if conn.weight < -1:
                conn.weight = -1
            elif conn.weight > 1:
                conn.weight = 1
    
        def new_connection(x):
            m = math.floor(random()*len(self.network_neurons))
            n = math.floor(random()*(len(self.network_neurons)-self.input_neurons))+self.input_neurons

            from_m = self.network_neurons[m]
            to_n = self.network_neurons[n]

            while n <= m or from_m.neuron_type==to_n.neuron_type or from_m.neuron_type.output:
                m = math.floor(random()*len(self.network_neurons))
                n = math.floor(random()*(len(self.network_neurons)-self.input_neurons))+self.input_neurons

                from_m = self.network_neurons[m]
                to_n = self.network_neurons[n]

            from_m = self.network_neurons[0]
            to_n = self.network_neurons[22]
            # test if a connection already exist' to this neuron
            for i in range(len(to_n.in_connections)):
                #print("i: " + str(i))
                if from_m==to_n.in_connections[i].from_n: 
                    #print("connection exists")
                    return
                if not to_n.neuron_type.output and to_n.neuron_index < to_n.in_connections[i].from_n.neuron_index:
                    #print("i dont think this is correct")
                    return
            for i in range(len(to_n.out_connections)):
                if from_m.neuron_index==to_n.in_connections[i].from_n.neuron_index: 
                    return
                if not to_n.out_connections[i].to_n.neuron_type.output and to_n.neuron_index > to_n.out_connections[i].to_n.neuron_index:
                    #print("i dont think this is correct")
                    return
            # add new connection from m to n
            # check if innovation exist'
            if str(from_m.neuron_index)+"->"+str(to_n.neuron_index) in self.innovation.found:
                #print("inno")
                #print(self.innovation.found[str(from_m.neuron_index)+"->"+str(to_n.neuron_index)])

                conn = Connection(
                                    from_m, # from neuron
                                    to_n, # to neuron
                                    (random()*2)-1, # connection weight
                                    self.innovation.found[str(from_m.neuron_index)+"->"+str(to_n.neuron_index)] # innovation number
                                    )

                self.network_connections.append(conn)
                from_m.out_connections.append(conn)
                to_n.in_connections.append(conn)

            else:
                #print("NO")
                conn = Connection(
                                    from_m, # from neuron
                                    to_n, # to neuron
                                    (random()*2)-1, # connection weight
                                    self.innovation.number # innovation number
                                    )

                self.innovation.found[str(from_m.neuron_index)+"->"+str(to_n.neuron_index)] = self.innovation.number
                self.innovation.number += 1
                self.network_connections.append(conn)
                from_m.out_connections.append(conn)
                to_n.in_connections.append(conn)

        def remove_connection(x): # 'x' is the given connection to remove from the network
            conn = self.network_connections[x]
            conn.from_n.out_connections.pop(conn.from_n.out_connections.index(conn))
            conn.to_n.in_connections.pop(conn.to_n.in_connections.index(conn))
            self.network_connections.pop(x)

        mutations = {0 : mutate_weight, 1 : new_connection, 2 : remove_connection}
        for n in range(20):
            print(self.innovation.found)
            mutations[1](val) # randomly select a mutation method
            mutations[2](200) # randomly select a mutation method
        

neural_network = NeuralNetwork(10, 10, 10, [], [])
neural_network.construct()
neural_network.mutate()


for i in range(neural_network.input_neurons,
               neural_network.network_size()
               ):

    if neural_network.network_neurons[i].neuron_type.output:
        neural_network.predict(neural_network.network_neurons[i].in_connections)


for i in range(neural_network.input_neurons+neural_network.output_neurons):
    print(neural_network.network_neurons[i].value)

print(len(neural_network.network_connections))

