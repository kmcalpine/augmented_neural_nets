import neural_network 
import agent
import argparse
from dataclasses import dataclass
from dataclasses import field
from typing import List

@dataclass
class NetworkHandler:
    population_a: List = field(default_factory=list)
    population_b: List = field(default_factory=list)

    def pr(self, a):

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

    def init_networks(self, population_size, innovations):
        for i in range(population_size):
            genome = neural_network.NeuralNetwork(5, 0, 5, [], [], innovations)
            genome.construct()
            pos = agent.Position(0, 0)
            a = agent.Agent(pos, genome)
            self.population_a.append(genome)

        for el in self.population_a:
            self.pr(el)



if __name__ == '__main__':
    global_innovations = neural_network.Innovations(0, {}) # initialize global innovations object
    network_handler = NetworkHandler([])
    network_handler.init_networks(3, global_innovations)
