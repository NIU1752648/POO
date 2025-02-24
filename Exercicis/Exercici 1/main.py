"""
Programació Orienteda a Objectes
Exercici 1: n-body problem
Autors: Celia, Gea, Nico
"""

import math
import numpy as np
import pygame

class Body:
    G = 6.67e-11 # Static Attribute

    def __init__(self, mass: float, initial_position: list, initial_velocity: list):
        """Initializes a new instance of a Body"""
        self._mass = mass # Private
        self._position = np.array(initial_position) # Public
        self._velocity = np.array(initial_velocity) # Public

    @property
    def mass(self):
        """Getter for private attribute mass"""
        return self._mass

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity


    @staticmethod
    def _distance(a: np.array, b: np.array) -> float:
        """Calculate the distance between two points"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def force_from(self, other) -> np.array:
        """Calculate the force acting on our object by another object"""
        try:
            d = Body._distance(self.position, other.position)
            return Body.G * ((self.mass * other.mass)/d**3) * (other.position - self.position)
        except ZeroDivisionError:
            return np.array((0, 0))

    def update(self, total_force: np.array, step: float):
        """Updates all vectors associated to Body"""
        self._velocity += (total_force/self.mass)*step
        self._position += self.velocity*step

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
        return self._num_bodies

    def _forces_acting_upon(self, body: Body) -> np.array:
        """Calculates the sum of all forces acting on a Body"""
        n_sum = np.array((0.0, 0.0))
        for other in self.bodies:
            n_sum += body.force_from(other)
        return n_sum

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
                    px, py, vx, vy, m = [float(z) for z in line.strip().split(' ') if z]
                    bodies.append(Body(m, [px, py], [vx, vy]))
        return cls(radius, bodies)

    def __str__(self):
        bs = '\n' # Backslash
        return f"Universe of radius: {self.radius}, formed by bodies:{bs}{bs.join(str(body) + ' Total forces: ' + str(self._forces_acting_upon(body)) for body in self.bodies)}"

class NBodySimulator:
    def __init__(self, step: float, filename: str, window_size = 600):
        self._step = step
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

    def _update(self):
        self.universe.update(self._step)
        self._time_passed += self._step
        self._t += 1

    def _rescale(self, point: np.array) -> np.array:
        r = self.universe.radius
        # Normalize to [0, 1] range
        normalized_x = (point[0] + r) / (2 * r)
        normalized_y = (r - point[1]) / (2 * r)  # Invert y-axis for Pygame
        # Scale to screen size
        screen_x = int(normalized_x * self.window_size)
        screen_y = int(normalized_y * self.window_size)
        return np.array([screen_x, screen_y])

    def fill_screen(self):
        self._screen.fill("purple")

    def _point(self, x: float, y:float, color, radius):
        r_x, r_y = self._rescale(np.array((x, y)))
        pygame.draw.circle(self._screen,
                           color,
                           (r_x, r_y),
                           radius)

    def draw(self):
        self._update()
        for body in self.universe.bodies:
            self._point(body.position[0], body.position[1], (255, 255, 255), 5)
        pygame.display.flip()

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
    try:
        sim = NBodySimulator(1000, "data/5body.txt")
    except FileNotFoundError:
        sim = NBodySimulator(1000, "Exercicis/Exercici 1/data/3body2.txt")


    sim.open_window()
    sim.fill_screen()
    force_stop = True

    while force_stop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                force_stop = False

        sim.fill_screen()
        sim.draw()

    print("Simulation Ended")

if __name__ == "__main__":
    main()