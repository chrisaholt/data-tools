import numpy as np

def perpendicular_to(v):
    """Computes a vector perpendicular to the input vector."""
    if v.shape[0] == 2:
        return np.array([-v[1], v[0]])
    else:
        raise ValueError("Only 2D vectors are supported.")


def convex_hull_2d(points: np.array):
    """Computes the convex hull of a set of 2D points."""

    # Encode the convex hull as a list of indices of the points.
    convex_hull_indices = []

    # First, compute a point which is definitely on the convex hull.
    # Do this by finding the mean of the points, then finding the point
    # which is furthest from the mean.
    mean_point = np.mean(points, axis=0)
    distances_to_mean = np.linalg.norm(points - mean_point, axis=-1)
    max_distance_index = int(np.argmax(distances_to_mean))
    convex_hull_indices.append(max_distance_index)

    latest_point = points[max_distance_index, :]
    point_to_mean = mean_point - latest_point
    normal = point_to_mean / np.linalg.norm(point_to_mean)
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