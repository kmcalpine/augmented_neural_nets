from dataclasses import dataclass
from dataclasses import field
from random import random
from random import randrange
from typing import Dict
import collections
import math

@dataclass
class Innovations:
    '''Class to represent the innovation number of 
       a given connection. This is a global object 
       that holds every unique connection found across
       of neural networks'''
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

    def mutate_weight(self):
        self.weight += ((random()*2)-1)*0.1
        if self.weight < -1:
            self.weight = -1
        elif self.weight > 1:
            self.weight = 1

@dataclass
class NeuralNetwork:
    input_neurons: int
    hidden_neurons: int
    output_neurons: int
    network_connections: [Connection]
    network_neurons: [Neuron]
    innovation: Innovations
    node_index: int = 0
    fitness_score: float = 0.0

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

                    in_num = self.innovation.found.get(str(i)+'->'+str(j))
                    if not in_num:
                        in_num = self.innovation.number
                        self.innovation.number += 1  
                        
                    conn = Connection(
                                        self.network_neurons[i], # from neuron
                                        self.network_neurons[j], # to neuron
                                        (random()*2)-1, # connection weight
                                        in_num # innovation number
                                        )

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


    def mutate(self):
        val = 5
        def mutate_weight():
            # randomly select a network connection and mutate its weight
            conn = self.network_connections[math.floor(random()*len(self.network_connections))]
            conn.mutate_weight()


        def create_new_connection(from_m, to_n, weight=(random()*2)-1):
            # check if innovation exist'
            if str(from_m.neuron_index)+"->"+str(to_n.neuron_index) in self.innovation.found:
                #print("inno")
                #print(self.innovation.found[str(from_m.neuron_index)+"->"+str(to_n.neuron_index)])

                conn = Connection(
                                    from_m, # from neuron
                                    to_n, # to neuron
                                    weight, # connection weight
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
                                    weight, # connection weight
                                    self.innovation.number # innovation number
                                    )

                self.innovation.found[str(from_m.neuron_index)+"->"+str(to_n.neuron_index)] = self.innovation.number
                self.innovation.number += 1
                self.network_connections.append(conn)
                from_m.out_connections.append(conn)
                to_n.in_connections.append(conn)
    
        def new_connection():
            m = math.floor(random()*len(self.network_neurons))
            n = math.floor(random()*(len(self.network_neurons)-self.input_neurons))+self.input_neurons

            from_m = self.network_neurons[m]
            to_m = self.network_neurons[n]

            while n <= m or from_m.neuron_type==to_m.neuron_type or from_m.neuron_type.output:
                m = math.floor(random()*len(self.network_neurons))
                n = math.floor(random()*(len(self.network_neurons)-self.input_neurons))+self.input_neurons

                from_m = self.network_neurons[m]
                to_m = self.network_neurons[n]

            # test if a connection already exist' to this neuron
            for i in range(len(to_m.in_connections)):
                #print("i: " + str(to_m.in_connections[i].innovation))
                if from_m is to_m.in_connections[i].from_n: 
                    return
                if not to_m.neuron_type.output and to_m.neuron_index < to_m.in_connections[i].from_n.neuron_index:
                    #print("i dont think this is correct")
                    return

            for i in range(len(to_m.out_connections)):
                if from_m.neuron_index==to_m.in_connections[i].from_n.neuron_index: 
                    return
                if not to_m.out_connections[i].to_m.neuron_type.output and to_m.neuron_index > to_m.out_connections[i].to_m.neuron_index:
                    #print("i dont think this is correct")
                    return
            # add new connection from m to n
           
            create_new_connection(from_m, to_m)

        def remove_connection(): # 'x' is the given connection to remove from the network

            if len(self.network_connections)==0: return
            x = math.floor(random()*len(self.network_connections))
            conn = self.network_connections[x]
            
            for i, from_conns in enumerate(conn.from_n.out_connections):
                
                if from_conns.innovation==conn.innovation:
                    print("innovation: " + str(from_conns.innovation))
                    conn.from_n.out_connections.pop(i)

            for i, to_conns in enumerate(conn.to_n.in_connections):
                
                if to_conns.innovation==conn.innovation:
                    print("innovation: " + str(to_conns.innovation))
                    conn.to_n.in_connections.pop(i)
            
            self.network_connections.pop(x)

        def check_conn_exists(from_m, to_m):
            for i in range(len(from_m.out_connections)-1, -1, -1):
                if from_m.out_connections[i].to_n==to_m:
                    return True
            return False

        def remove_neuron():
            '''Method to remove a neuron from the network'''
            if len(self.network_neurons)>self.input_neurons+self.output_neurons:
                x = math.floor(random()*len(self.network_neurons)-(self.input_neurons+self.output_neurons))
                neuron = self.network_neurons[x+self.input_neurons+self.output_neurons]

                neuron_inputs = []
                r_connections = [] # keep track of which connections will be removed
                for i in range(len(neuron.in_connections)-1, -1, -1):
                    neuron_inputs.append((neuron.in_connections[i].from_n, neuron.in_connections[i].weight))
                    r_connections.append(neuron.in_connections[i].innovation)
                neuron_outputs = []
                for i in range(len(neuron.out_connections)-1, -1, -1):
                    neuron_outputs.append((neuron.out_connections[i].to_n, neuron.out_connections[i].weight))
                    r_connections.append(neuron.out_connections[i].innovation)

                for i in range(len(neuron_inputs)-1, -1, -1):
                    for j in range(len(neuron_outputs)-1, -1, -1):
                        if not len(neuron_inputs[i][0].out_connections):                        
                            create_new_connection(neuron_inputs[i][0], neuron_outputs[j][0], neuron_inputs[i][1])
                            continue
                        else:
                            if not check_conn_exists(neuron_inputs[i][0], neuron_outputs[j][0]):
                                create_new_connection(neuron_inputs[i][0], neuron_outputs[j][0], neuron_inputs[i][1])
                
                for inno in r_connections:
                   for i, conn in enumerate(self.network_connections):
                       if inno==conn.innovation:
                           remove_connection(i)
                           break
                self.network_neurons.pop(x)
         
                
        def add_neuron():
            # randomly select a connection to insert a neuron
            idx = math.floor(random()*len(self.network_connections))
            conn = self.network_connections[idx]
            from_m = conn.from_n
            to_m = conn.to_n
            weight = conn.weight

            for i, out_conns in enumerate(from_m.out_connections):
                if out_conns.innovation==conn.innovation:
                    from_m.out_connections.pop(i)
                    break
            for i, in_conns in enumerate(to_m.in_connections):
                if in_conns.innovation==conn.innovation:
                    to_m.in_connections.pop(i)
                    break

            self.network_connections.pop(idx)

            n_type = NeuronType(0,1,0)
            n = Neuron(0, [], [], n_type, self.node_index)
            self.node_index += 1
            self.network_neurons.append(n)

            conn_from = create_new_connection(from_m, n, weight)
            conn_to = create_new_connection(n, to_m, 1)
            
        mutations = {0 : mutate_weight, 1 : new_connection, 2 : remove_connection, 3 : add_neuron, 4 : remove_neuron}
        mutation_operation = randrange(5) # randomly select mutation operation
        mutations[mutation_operation]() # apply mutation operation



    # fix crossover implementaion
    #
    #def crossover(parent1, parent2):
    #   offspring = NeuralNetwork(10, 0, 10, [], [])
    #
    #   matching_connections = []
    #   if parent1.fitness_score >= parent2.fitness_score:
    #       for i in range(len(parent1.network_connections)):
    #           for j in range(len(parent2.network_connections)):
    #               if parent1.network_connections[i].innovation==parent2.network_connections[j].innovation:
    #                   matching_connections.append(parent1.network_connections[i])
    #                   break
    #   else:
    #       for i in range(len(parent1.network_connections)):
    #           for j in range(len(parent2.network_connections)):
    #               if parent1.network_connections[i].innovation==parent2.network_connections[j].innovation:
    #                   matching_connections.append(parent1.network_connections[i])
    #                   break
    

#neural_network = NeuralNetwork(10, 0, 10, [], [])
#neural_network.construct()
#neural_network.mutate()
#
#
#for i in range(neural_network.input_neurons,
#               neural_network.network_size()
#               ):
#
#    if neural_network.network_neurons[i].neuron_type.output:
#        neural_network.predict(neural_network.network_neurons[i].in_connections)
#
#
#for i in range(neural_network.input_neurons+neural_network.output_neurons):
#    print(neural_network.network_neurons[i].value)
#
#print(len(neural_network.network_connections))