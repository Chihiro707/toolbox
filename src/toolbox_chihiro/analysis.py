import numpy as np
from tqdm import tqdm

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

def is_particle_in_element(
    nodes,
    elements,
    particle_element_id,
    lon,
    lat
    ):
    """
    Check whether a particle is inside a specified triangular element.

    Parameters
    ----------
    nodes : dict
        Dictionary of node_id to node coordinates.
    elements : array-like
        Array of elements defined by node indices.
    particle_element_id : int
        ID of the target element (1-based).
    lon : float
        Longitude of the particle.
    lat : float
        Latitude of the particle.

    Returns
    -------
    bool
        True if the particle is inside the element, False otherwise.
    """

    # Get node coordinates of the target element
    node_ids = elements[particle_element_id - 1][2:5]

    node_coords = np.array([
        [nodes[node_id]["lon"], nodes[node_id]["lat"]]
        for node_id in node_ids
    ])

    # Calculate barycentric coordinates
    p = np.array([lon, lat])

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

    # Check whether the triangle is degenerate
    if denominator <= np.finfo(float).eps * d00 * d11:
        return False

    beta = (d11 * d20 - d01 * d21) / denominator
    gamma = (d00 * d21 - d01 * d20) / denominator
    alpha = 1.0 - beta - gamma

    tolerance = 1e-10

    return (
        alpha >= -tolerance
        and beta >= -tolerance
        and gamma >= -tolerance
    )

def build_element_neighbors(elements):
    """
    Build adjacency relationships between triangular elements.
    Elements sharing an edge are considered neighbors.
    """

    edge_to_elements = {}

    for elem in elements:
        element_id = int(elem[0])
        node_ids = elem[2:5]

        edges = [
            tuple(sorted((node_ids[0], node_ids[1]))),
            tuple(sorted((node_ids[1], node_ids[2]))),
            tuple(sorted((node_ids[2], node_ids[0])))
        ]

        for edge in edges:
            edge_to_elements.setdefault(edge, []).append(element_id)

    element_neighbors = {
        int(elem[0]): set() for elem in elements
    }

    for adjacent_elements in edge_to_elements.values():
        if len(adjacent_elements) == 2:
            elem_a, elem_b = adjacent_elements

            element_neighbors[elem_a].add(elem_b)
            element_neighbors[elem_b].add(elem_a)

    return element_neighbors

def find_particle_element_nearby(
    nodes,
    elements,
    element_neighbors,
    previous_element_id,
    lon,
    lat
):
    """
    Search the previous element and its neighboring elements.
    """

    # Check the previous element first
    if is_particle_in_element(
        nodes, elements, previous_element_id, lon, lat
    ):
        return previous_element_id

    # Check neighboring elements
    for neighbor_id in element_neighbors[previous_element_id]:

        if is_particle_in_element(
            nodes, elements, neighbor_id, lon, lat
        ):
            return neighbor_id

    return None

def analyze_depth_residence(
    nodes,
    elements,
    lon,
    lat,
    depth,
    origin_marker,
    time_values,
    surface_layer_thickness=1.0,
    bottom_layer_thickness=1.0
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
    surface_layer_thickness : float, optional
        Thickness of the surface layer in meters. Default is 1.0.
    bottom_layer_thickness : float, optional
        Thickness of the bottom layer in meters. Default is 1.0.

    Returns
    -------
    dict
        Number of residence frames for each particle and depth layer.
    """

    marker_values = np.unique(origin_marker[:, 0])

    layers = ["surface", "in-water", "seabed", "out-of-domain"]

    # Group particles by origin marker
    marker_groups = {
        marker: np.where(origin_marker[:, 0] == marker)[0]
        for marker in marker_values
    }

    # Initialize residence counts and particle information
    residence_count = {}
    previous_element_id = {}
    is_out_of_domain = {}

    element_neighbors = build_element_neighbors(elements)

    for marker in marker_values:

        residence_count[marker] = {}
        previous_element_id[marker] = {}
        is_out_of_domain[marker] = {}

        for particle_id in marker_groups[marker]:

            residence_count[marker][particle_id] = {
                layer: 0 for layer in layers
            }

            previous_element_id[marker][particle_id] = None
            is_out_of_domain[marker][particle_id] = False

    # Analyze each frame
    for frame in tqdm(
        range(len(time_values)),
        desc="Analyzing particle residence",
        unit="frame"
    ):

        for marker in marker_values:

            for particle_id in marker_groups[marker]:

                # Skip particles already outside the domain
                if is_out_of_domain[marker][particle_id]:
                    residence_count[marker][particle_id]["out-of-domain"] += 1
                    continue

                # Check the previous element first
                particle_element_id = previous_element_id[marker][particle_id]

                if particle_element_id is not None:

                    particle_element_id = find_particle_element_nearby(
                        nodes,
                        elements,
                        element_neighbors,
                        particle_element_id,
                        lon[particle_id, frame],
                        lat[particle_id, frame]
                    )

                # Search all elements only if necessary
                if particle_element_id is None:

                    particle_element_id = find_particle_element(
                        nodes,
                        elements,
                        lon[particle_id, frame],
                        lat[particle_id, frame]
                    )
                # Count particles outside the domain
                if particle_element_id is None:
                    is_out_of_domain[marker][particle_id] = True
                    residence_count[marker][particle_id]["out-of-domain"] += 1
                    continue

                # Update the previous element ID
                previous_element_id[marker][particle_id] = particle_element_id

                # Classify particles into depth layers
                layer = judge_layer(
                    nodes,
                    elements,
                    particle_element_id,
                    depth[particle_id, frame],
                    surface_layer_thickness=surface_layer_thickness,
                    bottom_layer_thickness=bottom_layer_thickness
                )

                residence_count[marker][particle_id][layer] += 1

    return residence_count
