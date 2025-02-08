import numpy as np

class Hyperplane:
    """A class representing a hyperplane in arbitrary-dimensional space."""
    def __init__(self, point, normal):
        self._point = point
        self._normal = normal

    @property
    def point(self):
        return self._point
    
    @property
    def normal(self):
        return self._normal