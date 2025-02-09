import numpy as np
import plotly.graph_objects as go

from .hyperplane import Hyperplane

def create_hyperplane_3d_scatters(
        hyperplane: Hyperplane,
        normal_scale: float = 1.0,
        color: str = "rgba(0, 0, 255, 0.5)",    
    ):
    """Visualizes a hyperplane in 3D space."""
    scatters = []

    for point in hyperplane.points:
        assert point.shape == (3,), "Only 3D hyperplanes are supported."

        # Add plot of single point.
        scatters.append(
            go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode="markers",
                marker=dict(symbol="x", size=5, color=color),
                showlegend=False,
            )
        )

        # Add plot of normal vector.
        start = point
        end = start + normal_scale * hyperplane.normal
        scatters.append(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=color),
                showlegend=False,
            )
        )

    return scatters


