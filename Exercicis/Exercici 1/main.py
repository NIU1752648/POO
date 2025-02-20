"""
Programació Orienteda a Objectes
Exercici 1: n-body problem
Autors: Celia, Gea, Nico
"""

import numpy as np
import matplotlib.pyplot as mpl
import pygame

class Integrator: #Future implementation Runge-Kutta
    pass

class Body:
    G = 6.67e-11 # Static Attribute

    def __init__(self, mass: float, initial_position: list, initial_velocity: list):
        """Initializes a new instance of a Body"""
        self._mass = mass # Private
        self.position = np.array(initial_position) # Public
        self.velocity = np.array(initial_velocity) # Public

    @property
    def mass(self):
        """Getter for private attribute mass"""
        return self._mass

    @staticmethod
    def _distance(a, b) -> float:
        """Calculate the distance between two points"""
        pass #TODO

    def force_from(self, other) -> np.array:
        """Calculate the force acting on our object by another object"""
        pass #TODO

    def update(self, total_force: np.array, step: float):
        """Updates all vectors associated to Body"""
        pass #TODO

    def __str__(self):
        return f"Body of {self.mass} kg, positioned at {self.position} m and moving at {self.velocity} m/s"

class Universe:
    def __init__(self, radius: float, bodies: list):
        self.radius = radius
        self.bodies = bodies
        self._num_bodies = len(bodies)

    @property
    def num_bodies(self):
        """Getter for private attribute num_bodies"""
        return self._num_bodies

    @staticmethod
    def forces_acting_upon(body: Body) -> np.array:
        """Calculates the sum of all forces acting on a Body"""
        pass #TODO

    @classmethod
    def from_file(cls, filename):
        """Initialize a Universe instance with data from a file"""
        bodies = list()
        with open(filename, 'r') as file:
            num_bodies = int(file.readline())
            radius = float(file.readline())
            for x in range(num_bodies): # Iterates for each Body
                line = file.readline()
                if line != '':
                    m, px, py, vx, vy = [float(z) for z in line.strip().split(' ') if z]
                    bodies.append(Body(m, [px, py], [vx, vy]))
        return cls(radius, bodies)

    def __str__(self):
        bs = '\n' # Backslash
        return f"Universe of radius: {self.radius}, formed by bodies:{bs}{bs.join(str(body) for body in self.bodies)}"

class NBodySimulator:
    def __init__(self, step: float, step_num: int, filename: str, window_size = 600):
        self._step = step
        self._step_num = step_num
        self._t = float()
        self.universe = Universe.from_file(filename)
        self._window_size = window_size

    @property
    def window_size(self):
        return self._window_size

    def animate(self):
        pass

    def _draw_body(self):
        pass




if __name__ == "__main__":
    universe = Universe.from_file("data/3body2.txt")
    print(universe)