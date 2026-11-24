import numpy as np

def find_particle_element(nodes, elements, lon, lat):
    """
    Determines the element index containing a particle.

    Parameters
    ----------
    nodes : dict
        Dictionary of node_id to node coordinates.
    elements : array-like
        Array of elements defined by node indices.
    lon : float
        Longitude of the particle.
    lat : float
        Latitude of the particle.

    Returns
    -------
    int or None
        Element ID containing the particle, or None if the particle
        is outside the mesh.
    """

    p = np.array([lon, lat])

    for elem in elements:
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

        if denominator <= np.finfo(float).eps * d00 * d11:
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
            particle_element_id = int(elem[0])
            return int(particle_element_id)

    return None

def judge_layer(
    nodes,
    elements,
    particle_element_id,
    particle_depth,
    surface_layer_thickness=1.0,
    bottom_layer_thickness=1.0,
    ):
    """
    Classify a particle into surface, in-water, or seabed layer.

    Parameters
    ----------
    nodes : dict
        Dictionary of node_id to node information.
    elements : array-like
        Array of elements defined by node indices.
    particle_element_id : int
        ID of the element containing the particle (1-based).
    particle_depth : float
        Depth of the particle.
    surface_layer_thickness : float, optional
        Thickness of the surface layer [m].
    bottom_layer_thickness : float, optional
        Thickness of the seabed layer [m].

    Returns
    -------
    str
        "surface", "in-water", or "seabed".
    """

    if np.isnan(particle_element_id):
        layer = "out of domain"
        return layer

    # Validate layer thicknesses
    if surface_layer_thickness < 0 or bottom_layer_thickness < 0:
        raise ValueError("Layer thicknesses must be non-negative.")

    # Get node IDs belonging to the particle's element
    node_ids = elements[particle_element_id - 1][2:5]

    # Calculate the average depth of the element
    element_depths = np.array([
        abs(nodes[node_id]["depth"]) for node_id in node_ids
    ])
    depth_of_element = np.mean(element_depths)

    # Check whether the layers overlap
    if depth_of_element <= surface_layer_thickness + bottom_layer_thickness:
        raise ValueError("Layer thickness is too big for this element.")

    # Classify the particle
    particle_depth = abs(particle_depth)

    if particle_depth < surface_layer_thickness:
        layer = "surface"

    elif particle_depth < depth_of_element - bottom_layer_thickness:
        layer = "in-water"

    else:
        layer = "seabed"

    return layer

def analyze_depth_residence(
    nodes,
    elements,
    lon,
    lat,
    depth,
    origin_marker,
    time_values
    ):
    """
    Analyze the number of frames particles spend in each depth layer.

    Parameters
    ----------
    nodes : dict
        Dictionary of node_id to node information.
    elements : array-like
        Array of elements defined by node indices.
    lon, lat, depth : numpy.ndarray
        Particle longitude, latitude, and depth.
        Shape: (n_particles, n_times).
    origin_marker : numpy.ndarray
        Marker identifying the particle seeding locations.
    time_values : array-like
        Time values corresponding to each frame.

    Returns
    -------
    dict
        Number of residence frames for each particle and depth layer.
    """

    marker_values = np.unique(origin_marker[:, 0])

    layers = ["surface", "in-water", "seabed", "out-of-domain"]

    # Initialize residence counts
    residence_count = {}

    for marker in marker_values:

        marker_group = np.where(origin_marker[:, 0] == marker)[0]

        residence_count[marker] = {
            particle_id: {layer: 0 for layer in layers}
            for particle_id in marker_group
        }

    # Analyze each frame
    for frame in range(len(time_values)):

        for marker in marker_values:

            marker_group = np.where(origin_marker[:, 0] == marker)[0]

            for particle_id in marker_group:

                # Find the element containing the particle
                particle_element_id = find_particle_element(
                    nodes,
                    elements,
                    lon[particle_id, frame],
                    lat[particle_id, frame]
                )

                # Count particles outside the domain
                if particle_element_id is None:
                    residence_count[marker][particle_id]["out-of-domain"] += 1
                    continue

                # Classify particles into depth layers
                layer = judge_layer(
                    nodes,
                    elements,
                    particle_element_id,
                    depth[particle_id, frame],
                    surface_layer_thickness=1.0,
                    bottom_layer_thickness=1.0
                )

                residence_count[marker][particle_id][layer] += 1

    return residence_count

                
