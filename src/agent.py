import neural_network 
from dataclasses import dataclass
from dataclasses import field

@dataclass
class Position:
    x: float
    y: float 

@dataclass
class Agent:
    position: Position
    brain: neural_network