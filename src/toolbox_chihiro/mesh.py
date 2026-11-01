import numpy as np

def read_mesh(file_path):
    """
    Reads a mesh file in the specified format
    and returns the nodes, elements, open boundary nodes, and land boundary nodes.
    """

    with open (file_path, 'r') as f:

        # Skip the first line (header)
        f.readline()

        # Read the number of elements and nodes
        temp = f.readline()
        n_elements = int(temp.split()[0])
        n_nodes = int(temp.split()[1])

        # Nodes
        nodes = np.array([
            [float(value) for value in f.readline().split()]
            for _ in range(n_nodes)
            ])

        # Elements
        elements = np.array([
            [float(value) for value in f.readline().split()]
            for _ in range(n_elements)
            ])

        # Open boundaries
        n_open_boundaries = int(f.readline().split()[0])
        f.readline() # Skip the line (the number of total open boundary nodes)

        open_boundary_nodes = []

        for _ in range(n_open_boundaries):

            n_open_boundary_nodes = int(f.readline().split()[0])
            boundary_nodes = np.array([
                int(f.readline().strip())
                for _ in range(n_open_boundary_nodes)
            ])

            open_boundary_nodes.append(boundary_nodes)

        # Land boundaries
        n_land_boundaries = int(f.readline().split()[0])
        f.readline() # Skip the line (the number of total land boundary nodes)

        land_boundary_nodes = []

        for _ in range(n_land_boundaries):

            n_land_boundary_nodes = int(f.readline().split()[0])
            boundary_nodes = np.array([
                int(f.readline().strip())
                for _ in range(n_land_boundary_nodes)
            ])

            land_boundary_nodes.append(boundary_nodes)

    return (nodes, elements, open_boundary_nodes, land_boundary_nodes)