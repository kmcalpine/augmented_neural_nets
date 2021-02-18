import neural_network 
import math
from random import random

class Position:
    def __init__(self) -> None:
        self.x = math.floor(random()*(random()*100))
        self.y = math.floor(random()*(random()*100))
        
class Resource:
    def __init__(self, p: Position):
        self.position = p

class Agent:
    def __init__(self, p: Position, b: neural_network) -> None:
        self.position = p
        self.brain = b
        self.species = None
        self.distance_resource = None
        self.distance_agent = None
        self.angle_resource = None
        self.angle_agent = None
        self.energy = 100
        self.dead = False