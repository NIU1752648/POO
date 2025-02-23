"""
Programació Orienteda a Objectes
Exercici 1: n-body problem
Autors: Celia, Gea, Nico
"""

import math
import numpy as np
#import matplotlib.pyplot as mpl
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
    def _distance(a: np.array, b: np.array) -> float:
        """Calculate the distance between two points"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def force_from(self, other) -> np.array:
        """Calculate the force acting on our object by another object"""
        try:
            d = Body._distance(self.position, other.position)
            return Body.G * ((self.mass * other.mass)/d**2) * ((other.position - self.position)/d)
        except ZeroDivisionError:
            return np.array((0, 0))

    def update(self, total_force: np.array, step: float):
        """Updates all vectors associated to Body"""
        self.velocity += (total_force/self.mass)*step
        self.position += self.velocity*step

    def __str__(self):
        return f"Body of {self.mass} kg, positioned at {self.position} m and moving at {self.velocity} m/s"

class Universe:
    def __init__(self, radius: float, bodies: list):
        self._radius = radius
        self._bodies = bodies
        self._num_bodies = len(bodies)

    @property
    def radius(self):
        return self._radius

    @property
    def bodies(self):
        return self._bodies

    @property
    def num_bodies(self):
        """Getter for private attribute num_bodies"""
        return self._num_bodies

    def _forces_acting_upon(self, body: Body) -> np.array:
        """Calculates the sum of all forces acting on a Body"""
        sum = np.array((0.0, 0.0))
        for other in self.bodies:
            sum += body.force_from(other)
        return sum

    def update(self, step: float):
        for body in self.bodies:
            body.update(self._forces_acting_upon(body), step)

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
        self._time_passed = float()
        self._t = int()
        self.universe = Universe.from_file(filename)
        self._window_size = window_size
        self._screen = None
        self._clock = None

    @property
    def window_size(self):
        return self._window_size

    def open_window(self):
        pygame.init()
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

    def sim_ended(self) -> bool:
        return self._t == self._step_num

    def _update(self):
        self.universe.update(self._step)
        self._time_passed += self._step
        self._t += 1

    def _rescale(self, point: np.array) -> np.array:
        r = self.universe.radius
        return self.window_size * np.array(
            ((point[0] - r) / (2 * r),
            (r - point[1]) / (2 * r))
        )

    def fill_screen(self):
        self._screen.fill("purple")

    def _point(self, x: float, y:float):
        n_point = self._rescale(np.array((x, y)))
        r_x = n_point[0]
        r_y = n_point[1]
        pygame.draw.circle(self._screen,
                           (0, 0, 0),
                           (r_x, r_y),
                           0.125)

    def draw(self):
        #TODO: Does not draw
        for body in self.universe.bodies:
            self._point(body.position[0], body.position[1])
        self.universe.update(self._step)
        pygame.display.flip()
        self._clock.tick(60)

    #TEMP
    def log_state(self):
        """Logs universe state at current time in log.txt"""
        if self._t == 0:
            with open("log.txt", "w") as file:
                file.write(f"At t = {self._t} and {self._time_passed}s\n{str(self.universe)}\n")
        else:
            with open("log.txt", "a") as file:
                file.write(f"\nAt t = {self._t} and {self._time_passed}s\n{str(self.universe)}\n")


def main():
    sim = NBodySimulator(0.001, 100, "data/3body2.txt")

    sim.open_window()

    force_stop = True

    while not (sim.sim_ended()) and force_stop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                force_stop = False

        sim.fill_screen()
        sim.log_state()
        sim.draw()

    print("Simulation Ended")
    # If simulation ends, don't close window
    running = True
    while running and force_stop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

if __name__ == "__main__":
    main()