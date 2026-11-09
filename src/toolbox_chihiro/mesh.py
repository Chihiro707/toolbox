import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xarray as xr
import pandas as pd

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

def draw_coastline(nodes, open_boundary_nodes, land_boundary_nodes):
    """
    Makes a coastal line plot from the mesh data.
    Returns a matplotlib figure and axis.

    Parameters
    ----------
    nodes : dict
        Dictionary of node_id to (lon, lat, depth).
    open_boundary_nodes : list
        List of arrays containing IDs of open boundary nodes.
    land_boundary_nodes : list
        List of arrays containing IDs of land boundary nodes.

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axis.
    """

    boundaries = open_boundary_nodes + land_boundary_nodes

    fig, ax = plt.subplots(figsize=(12, 10))

    for boundary_nodes in boundaries:
        lon = [nodes[n]["lon"] for n in boundary_nodes]
        lat = [nodes[n]["lat"] for n in boundary_nodes]

        ax.plot(lon, lat, "-k")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    return fig, ax

def animate_particles(
    nodes,
    open_boundary_nodes,
    land_boundary_nodes,
    lon,
    lat,
    origin_marker,
    time_values
):
    """
    Creates an animation of particle tracking results.
    """

    fig, ax = draw_coastline(
        nodes,
        open_boundary_nodes,
        land_boundary_nodes
    )

    title = ax.set_title("Time: ")

    scatter = ax.scatter(
        [], [],
        c=[],
        cmap="viridis",
        s=10,
        alpha=0.7
    )

    # 軸の範囲を固定
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    def update(frame):
        current_time = time_values[frame]

        # タイトルの更新
        title.set_text(f"Time: {current_time}")

        # 粒子の座標を更新
        positions = np.column_stack[lon[:, frame], lat[:, frame]]
        scatter.set_offsets(positions)

        # 粒子の色を更新
        scatter.set_array(origin_marker[:, frame])

    ani = FuncAnimation(
        fig,
        update,
        frames=len(time_values),
        interval=200,
        blit=False
    )

    return ani