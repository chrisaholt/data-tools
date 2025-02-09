import numpy as np

class Hyperplane:
    """A class representing a hyperplane in arbitrary-dimensional space."""
    def __init__(self, points, normal):
        self._points = points
        self._normal = normal

    @property
    def points(self):
        return self._points
    
    @property
    def normal(self):
        return self._normal