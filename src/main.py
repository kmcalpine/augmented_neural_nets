import neural_network 
import agent
import math
import copy
import argparse
from typing import List
from random import random, randrange

class NetworkHandler:
    def __init__(self, size):
        self.network_size = size
        self.population_a = []
        self.population_b = []
        self.resources = []
        self.count = 0
        self._MUTATION_RATE = 0.33

    def pr(self, a):
        # function to print cyclic objects
        # src: https://stackoverflow.com/questions/22736557/printing-cyclic-data-structures

        seen = set()
        def f(a, indent):
            for k, v in sorted(a.__dict__.items()):
                s = indent * ' '
                if type(v) == argparse.Namespace:
                    if id(v) in seen:
                        print(s, k)
                    else:
                        seen.add(id(v))
                        print(s, k)
                        f(v, indent + 1)
                elif type(v) == bytearray:
                    print(s, k, len(v))
                elif type(v) == list:
                    print(s, k, len(v))
                elif type(v) == str:
                    print(s, k, repr(v))
                elif type(v) == tuple:
                    print(s, k, len(v))
                elif type(v) in (int, float):
                    print(s, k, v)
                else:
                    print(s, k, v)
        f(a, 1)

    def init_networks(self, population_size, hidden, innovations):
        for i in range(population_size):
            genome = neural_network.NeuralNetwork(6, hidden, 8, [], [], innovations, 0, 0)
            genome.construct()
            pos = agent.Position()
            a = agent.Agent(pos, genome)
            if not hidden:
                self.population_a.append(a)
            else:
                self.population_b.append(a)

    def init_resources(self, population_size):
        for i in range(population_size):
            pos = agent.Position()
            resource = agent.Resource(pos)
            self.resources.append(resource)

    def init_distances(self) -> None:
        '''Initialize agent distance to resource and other agent'''
        for i in range(self.network_size):

           a = self.population_a[i]
           b = self.population_b[i]

           res = self.resources[i]
           
           a.distance_resource = self.get_distance(a, res)
           b.distance_resource = self.get_distance(b, res)

           a.distance_agent = self.get_distance(a, b)
           b.distance_agent = self.get_distance(b, a)


    def get_distance(self, a, b) -> float:
        '''Calculate and return distance.'''
        dist = math.sqrt(   (a.position.x - b.position.x) *
                            (a.position.x - b.position.x) +
                            (a.position.y - b.position.y) *
                            (a.position.y - b.position.y))

        return round(dist, 2)


    def init_angles(self) -> None:
        '''Initialize agent angles to resource and other agent'''
        for i in range(self.network_size):
           a = self.population_a[i]
           b = self.population_b[i]
           res = self.resources[i]
           
           a.angle_resource = self.get_angle(a, res)
           b.angle_resource = self.get_angle(b, res)

           a.angle_agent = self.get_angle(a, b)
           b.angle_agent = self.get_angle(b, a)

    def get_angle(self, a, b) -> float:
        '''Calculate and return angle.'''
        angle = math.atan2(a.position.y - b.position.y, a.position.x - b.position.x)
        angle = (angle / (math.pi*2)*360)+180

        return round(angle, 2)

    def normalize_inputs(self, inputs) -> [float]:
        input_min = min(inputs)
        input_max = max(inputs)
        range = input_max - input_min

        new_inputs = []

        for input in inputs:
            new_inputs.append((input-input_min)/range)

        return new_inputs



    def reset_network_neuron_values(self, inputs, genome) -> None:
        '''Reset values calulated from previous iteration'''
        for neuron in genome.network_neurons:
            if neuron.neuron_type.input:
                neuron.value = inputs.pop()
            else:
                neuron.value = 0
  
    def get_network_output(self, genome) -> int:
        '''Return maximum output index'''
        outputs = []
        for output_neuron in genome.network_neurons:
            if output_neuron.neuron_type.output:
                genome.predict(output_neuron.in_connections) # propogate inputs through network
                outputs.append(output_neuron.value)
        
        if not outputs:
            return -1
        return max(range(len(outputs)), key=lambda i: outputs[i])
        

    def action_update(self, action, _agent) -> None:
        '''Apply network output to agent'''
        _agent.energy -= 1;

        if action == -1:
            return

        def action_0():
            if _agent.position.x < 100:
                _agent.position.x += 1

        def action_1():
            if _agent.position.x < 100 and _agent.position.y < 100:
                _agent.position.x += (math.sqrt(2)/2)
                _agent.position.y += (math.sqrt(2)/2)

        def action_2():
            if _agent.position.x > 0:
                _agent.position.x -= 1

        def action_3():
            if _agent.position.x > 0 and _agent.position.y < 100:
                _agent.position.x -= (math.sqrt(2)/2)
                _agent.position.y += (math.sqrt(2)/2)

        def action_4():
            if _agent.position.y < 100:
                _agent.position.y += 1

        def action_5():
            if _agent.position.x < 100 and _agent.position.y > 0:
                _agent.position.x += (math.sqrt(2)/2)
                _agent.position.y -= (math.sqrt(2)/2)

        def action_6():
            if _agent.position.y > 0:
                _agent.position.y -= 1

        def action_7():
            if _agent.position.x > 0 and _agent.position.y > 0:
                _agent.position.x -= (math.sqrt(2)/2)
                _agent.position.y -= (math.sqrt(2)/2)

        #maps to the 8 possible network outputs, checks if within bounds of domain size
        action_switch = {
            0: action_0,
            1: action_1,
            2: action_2,
            3: action_3,
            4: action_4,
            5: action_5,
            6: action_6,
            7: action_7,
        }

        _action = action_switch.get(action)
        _action()


    def update_fitness(self, a, b, res) -> None:
        '''Calculate current fitness score of genome based on last action.'''
        dist_agent = self.get_distance(a, b)      
        dist_resource = self.get_distance(a, res)

        if dist_resource-0.01 > a.distance_resource:
            a.brain.fitness_score -= 150
            
        if dist_resource+0.1 < a.distance_resource:
            
            a.brain.fitness_score += 0.9
        if dist_resource <= 4:
            a.brain.fitness_score += 150
            a.energy += 100
            pos = agent.Position()
            res.position = pos
        if a.brain.fitness_score < 0:
            a.brain.fitness_score = 0
        

    def update(self, _agent) -> None:
        '''Methods to initialize network inputs.'''
        network_inputs = [
                _agent.distance_resource,
                _agent.distance_agent,
                _agent.angle_resource,
                _agent.angle_agent,
                _agent.position.x,
                _agent.position.y
            ]


        input_min = min(network_inputs)
        input_max = max(network_inputs)
        range = input_max - input_min

        new_inputs = []

        for input in network_inputs:
            new_inputs.append((input-input_min)/range)

        n = self.normalize_inputs(network_inputs)

        self.reset_network_neuron_values(new_inputs, _agent.brain)
        action = self.get_network_output(_agent.brain)
        self.action_update(action, _agent)

    def reset_positions(self) -> None:
        for _agent in self.population_a:
            pos = agent.Position()
            _agent.position = pos

        for _agent in self.population_b:
            pos = agent.Position()
            _agent.position = pos

        for res in self.resources:
            pos = agent.Position()
            res.position = pos

    def calculate_population(self) -> None:
        '''Method to call the respective update functions for agents'''
        if self.count == self.network_size*2:
            return

        for i in range(len(self.population_a)):
            agent_a = self.population_a[i]
            agent_b = self.population_b[i]
            resource = self.resources[i]

            if agent_a.energy > 0:
                self.update(agent_a)
                self.update_fitness(agent_a, agent_b, resource)
            else:
                self.count += 1

            if agent_b.energy > 0:
                self.update(agent_b)
                self.update_fitness(agent_b, agent_a, resource)
            else:
                self.count += 1

        self.init_distances()
        self.init_angles()


    def sort_fitness(self) -> None:
        '''Method to sort population of genomes by fitness scores
           allowing top % performers to be selected for passing to next generation   
        '''
        self.population_a.sort(key=lambda x: x.brain.fitness_score, reverse=True)
        self.population_b.sort(key=lambda x: x.brain.fitness_score, reverse=True)

    def generate_new_fixed_genomes(self) -> None:
        for _agent in self.population_b:
            _agent.energy = 100
            _agent.dead = False
            _agent.brain.fitness_score = 0
            _agent.brain.mutate_weight()
        
    def generate_new_genomes(self) -> None:
        new_pop = []
        for i in range(len(self.population_a)):
            choose = randrange(math.floor(len(self.population_a)*0.2)) # randomly choose from top 20%
            dc = copy.deepcopy(self.population_a[choose])

            r = random()
            if r < self._MUTATION_RATE:
                dc.brain.mutate() #apply mutation to new genome

            #dc.brain.crosover() -> bugged

            #create new genome
            genome = neural_network.NeuralNetwork(6, dc.brain.hidden_neurons,
                                                  8, dc.brain.network_connections,
                                                  dc.brain.network_neurons,
                                                  dc.brain.innovation,
                                                  dc.brain.node_index, 0)

            pos = agent.Position()
            a = agent.Agent(pos, genome)
            new_pop.append(a)

        #initialize population with next generation of genomes
        self.population_a = new_pop
            

if __name__ == '__main__':
    _NETWORK_SIZE = 20
    _GENERATIONS = 100

    global_innovations = neural_network.Innovations(0, {}) # initialize global innovations object
    network_handler = NetworkHandler(_NETWORK_SIZE)
    network_handler.init_networks(_NETWORK_SIZE, 0, global_innovations) #create evolving networks
    network_handler.init_networks(_NETWORK_SIZE, 5, global_innovations) #create fixed networks
    network_handler.init_resources(_NETWORK_SIZE) #create resources
    network_handler.init_distances()
    network_handler.init_angles()
    

    for i in range(_GENERATIONS):
        while network_handler.count < _NETWORK_SIZE*2:
            network_handler.calculate_population()

        network_handler.count = 0
        network_handler.sort_fitness()

        for i in range(len(network_handler.population_a)):
            print(network_handler.population_a[i].brain.fitness_score, network_handler.population_b[i].brain.fitness_score)

        network_handler.generate_new_genomes() #generate new population of genomes
        network_handler.generate_new_fixed_genomes() #generate new population of fixed size genomes

        network_handler.init_distances()
        network_handler.init_angles()
