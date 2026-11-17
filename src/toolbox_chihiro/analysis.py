import numpy as np

def find_particle_element(nodes, element, lon, lat):
    """
    Determines the element index containing a particle.

    Parameters
    ----------
    nodes : dict
        Dictionary of node_id to node coordinates.
    element : array-like
        Array of elements defined by node indices.
    lon : float
        Longitude of the particle.
    lat : float
        Latitude of the particle.

    Returns
    -------
    int
        Index of the element containing the particle.

    Raises
    ------
    ValueError
        If no element contains the particle.
    """

    p = np.array([lon, lat])

    for i, elem in enumerate(element):
        node_coords = np.array([
            [nodes[node_id]["lon"], nodes[node_id]["lat"]]
            for node_id in elem[2:5]
        ])

        a, b, c = node_coords

        v0 = b - a
        v1 = c - a
        v2 = p - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denominator = d00 * d11 - d01**2

        if np.isclose(denominator, 0):
            continue

        beta = (d11 * d20 - d01 * d21) / denominator
        gamma = (d00 * d21 - d01 * d20) / denominator
        alpha = 1 - beta - gamma

        tolerance = 1e-10

        if (
            alpha >= -tolerance
            and beta >= -tolerance
            and gamma >= -tolerance
        ):
            return i

    raise ValueError("No element contains the particle.")

# def analyze_depth_residence(
#         nodes,
#         lon,
#         lat,
#         depth,
#         origin_marker,
#         time_values,
#         surface_layer_thickness=1.0,
#         bottom_layer_thickness=1.0):
#     """
#     Analyzes the depth residence of particles over time.

#     Parameters
#     ----------
#     nodes : dict
#         Dictionary of node_id to (lon, lat, depth).
#     lon : array-like
#         Array of longitudes of particles.
#     lat : array-like
#         Array of latitudes of particles.
#     depth : array-like
#         Array of depths of particles.
#     origin_marker : array-like
#         Array indicating the origin marker for each particle.
#     time_values : array-like
#         Array of time values corresponding to the particle positions.

#     Returns
#     -------
#     None
#     """

#     for i in range(len(time_values)):