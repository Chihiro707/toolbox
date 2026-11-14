import numpy as np
import pandas as pd
import xarray as xr

def read_mesh(file_path):
    """
    Reads a mesh file in the specified format
    and returns the nodes, elements, open boundary nodes, and land boundary nodes.

    paramiters
    ----------
    file_path : str
        Path to the mesh file.

    Returns
    -------
    tuple
        A tuple containing:
        - nodes: dict of node_id to (lon, lat, depth)
        - elements: list of elements
        - open_boundary_nodes: list of numpy arrays for each open boundary
        - land_boundary_nodes: list of numpy arrays for each land boundary
    """

    with open (file_path, 'r') as f:

        # Skip the first line (header)
        f.readline()

        # Read the number of elements and nodes
        temp = f.readline()
        n_elements = int(temp.split()[0])
        n_nodes = int(temp.split()[1])

        # Nodes
        nodes = {
            int(node_id): {
                "lon": float(lon),
                "lat": float(lat),
                "depth": float(depth)
            }
            for node_id, lon, lat, depth in (
                f.readline().split() for _ in range(n_nodes)
            )
        }

        # Elements
        elements = [
            [float(value) for value in f.readline().split()]
            for _ in range(n_elements)
            ]

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

def read_particle(nc_path):
    """
    Reads particle data from a NetCDF file and
    returns the longitude, latitude, and time values.

    Parameters
    ----------
    nc_path : str
        Path to the NetCDF file.

    Returns
    -------
    tuple
        A tuple containing:
        - lon: numpy array of longitudes
        - lat: numpy array of latitudes
        - depth: numpy array of depths
        - origin_marker: numpy array of origin markers
        - time_values: pandas datetime index of time values
    """

    ds = xr.open_dataset(nc_path)

    lon = ds['lon'].to_numpy()
    lat = ds['lat'].to_numpy()
    depth = ds['z'].to_numpy()
    origin_marker = ds['origin_marker'].to_numpy()
    time_values = pd.to_datetime(ds['time'].to_numpy())

    ds.close()

    return lon, lat, depth, origin_marker, time_values
