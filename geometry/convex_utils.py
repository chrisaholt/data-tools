from typing import List, Tuple
import numpy as np
from .hyperplane import Hyperplane

def perpendicular_to(v: np.array) -> np.array:
    """Computes a vector perpendicular to the input vector."""
    if v.shape[0] == 2:
        return np.array([-v[1], v[0]])
    else:
        raise ValueError("Only 2D vectors are supported.")

def find_cosine_of_largest_angle_from_point_and_direction(
    base_point: np.array,
    direction: np.array,
    points: np.array,
    indices_to_ignore: List[int] = [],
) -> Tuple[float, int]:
    """Finds the point with the largest angle to the direction."""
    base_to_points = points - base_point
    norms = np.linalg.norm(base_to_points, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    base_to_points_dirs = base_to_points / norms
    cos_angles_to_dir = np.sum(base_to_points_dirs * direction, axis=-1)
    for idx in indices_to_ignore:
        cos_angles_to_dir[idx] = 1.0
    index_of_largest_angle = int(np.argmin(cos_angles_to_dir))
    cos_angle = cos_angles_to_dir[index_of_largest_angle]
    return cos_angle, index_of_largest_angle


def supporting_hyperplane_from_points(points: np.array) -> Tuple[Hyperplane, int]:
    """Finds a supporting hyperplane from a set of points."""
    mean_point = np.mean(points, axis=0)
    distances_to_mean = np.linalg.norm(points - mean_point, axis=-1)
    max_distance_index = int(np.argmax(distances_to_mean))
    max_distance_point = points[max_distance_index, :]
    point_to_mean = mean_point - max_distance_point
    normal = point_to_mean / np.linalg.norm(point_to_mean)
    return (
        Hyperplane([max_distance_point], normal),
        max_distance_index,
    )


def supporting_face_from_points(points: np.array) -> Tuple[Hyperplane, List[int]]:
    """Finds a supporting face from a set of points."""
    
    # Start with an initial plane, then iteratively add points to the face.
    hyperplane_point_indices = []
    hyperplane, latest_index = supporting_hyperplane_from_points(points)
    hyperplane_point_indices.append(latest_index)

    # Find the next point to add to the face by finding the point
    # with the largest angle to the normal of the current hyperplane.
    while True:
        if len(hyperplane_point_indices) >= 2:
            break  # TEMPORARY

        if len(hyperplane_point_indices) > len(points):
            raise ValueError("Supporting face computation failed.")
        
        # For each base point on the hyperplane, find the point with the largest angle to the normal.
        cos_angles = []
        furthest_point_indices = []
        for point in hyperplane.points:
            cos_angle, furthest_point_index = find_cosine_of_largest_angle_from_point_and_direction(
                point, hyperplane.normal, points, hyperplane_point_indices
            )
            cos_angles.append(cos_angle)
            furthest_point_indices.append(furthest_point_index)

        # Determine which point to add to the face.
        hyperplane_point_index_for_largest_angle = np.argmin(cos_angles)
        point_on_hyperplane = hyperplane.points[hyperplane_point_index_for_largest_angle]
        latest_index = furthest_point_indices[hyperplane_point_index_for_largest_angle]
        latest_point = points[latest_index, :]
        hyperplane_point_indices.append(latest_index)

        # Rotate the hyperplane normal so that now this latest_point is on the new hyperplane.
        point_on_hyperplane_to_latest_dir = latest_point - point_on_hyperplane
        point_on_hyperplane_to_latest_dir = point_on_hyperplane_to_latest_dir / np.linalg.norm(point_on_hyperplane_to_latest_dir)

        new_normal = hyperplane.normal - np.dot(hyperplane.normal, point_on_hyperplane_to_latest_dir) * point_on_hyperplane_to_latest_dir
        new_normal = new_normal / np.linalg.norm(new_normal)
        new_hyperplane_points = hyperplane.points + [latest_point]
        hyperplane = Hyperplane(new_hyperplane_points, new_normal)

    return hyperplane, hyperplane_point_indices

def convex_hull_2d(points: np.array):
    """Computes the convex hull of a set of 2D points."""

    # Encode the convex hull as a list of indices of the points.
    convex_hull_indices = []

    # First, compute a point which is definitely on the convex hull.
    # Do this by finding the mean of the points, then finding the point
    # which is furthest from the mean.
    latest_point, normal, max_distance_index = (
        supporting_hyperplane_from_points(points)
    )
    convex_hull_indices.append(max_distance_index)
    direction_of_line = perpendicular_to(normal)

    # The line with normal is a (not necessarily tight) supporting hyperplane of the 
    # convex hull at latest_point. We use this to find the next point by finding
    # which point has the smallest angle relative to the direction of the line
    latest_index = max_distance_index
    while True:
        if len(convex_hull_indices) > len(points):
            raise ValueError("Convex hull computation failed.")
        latest_to_points = points - latest_point
        norms = np.linalg.norm(latest_to_points, axis=-1, keepdims=True)
        norms[latest_index, :] = 1.0
        latest_to_points_dirs = (
            latest_to_points / norms
        )
        cos_angles_to_line = np.sum(latest_to_points_dirs * direction_of_line, axis=-1)
        cos_angles_to_line[latest_index] = -1.0
        latest_index = int(np.argmax(cos_angles_to_line))

        # If we've come back to the start, we're done.
        if latest_index == max_distance_index:
            break

        # Set the latest point to the next point and iterate.
        convex_hull_indices.append(latest_index)
        latest_point = points[latest_index, :]
        direction_of_line = latest_to_points_dirs[latest_index, :]

    return convex_hull_indices