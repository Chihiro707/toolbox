import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
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

def animate_hmove(
        nodes,
        open_boundary_nodes,
        land_boundary_nodes,
        lon,
        lat,
        origin_marker,
        time_values,
        interval=50,
        label=None
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

    # Get unique origin markers for color mapping
    marker_values = np.unique(origin_marker)
    expected_values = np.arange(len(marker_values))

    if not np.array_equal(marker_values, expected_values):
        raise ValueError(
            "Origin markers should be consecutive integers starting from 0. "
            f"Found: {marker_values}, expected: {expected_values}"
        )

    # Set labels
    if label is None:
        label = [str(value) for value in marker_values]

    if len(label) != len(marker_values):
        raise ValueError(
            "origin_marker must have the same length as label. "
            f"origin_marker: {len(marker_values)}, label: {len(label)}"
            )

    # Create a colormap
    cmap = plt.get_cmap("viridis", len(label))

    scatter = ax.scatter(
        [], [],
        c=[],
        cmap=cmap,

        # Fix the color limits to prevent the colormap from changing as
        # the number of particles increases or decreases.
        vmin=-0.5,
        vmax=len(label) - 0.5,

        s=10,
        alpha=0.7
    )

    handles = [
        Line2D(
            [0],[0],
            marker='o',
            color='w',
            markerfacecolor=cmap(i),
            markersize=7,
            label=name
        )
        for i, name in enumerate(label)
    ]

    ax.legend(handles=handles, title="Seeding Location")

    # Set the limits of the plot to the current limits
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    def update(frame):
        current_time = time_values[frame]

        # Update the title with the current time
        title.set_text(f"Time: {current_time}")

        # Update the positions of the particles
        positions = np.column_stack((lon[:, frame], lat[:, frame]))
        scatter.set_offsets(positions)

        # Update the colors of the particles based on their origin markers
        scatter.set_array(origin_marker[:, frame])

    ani = FuncAnimation(
        fig,
        update,
        frames=len(time_values),
        interval=interval,
        blit=False
    )

    return ani


def animate_vmove(
    lon,
    depth,
    origin_marker,
    time_values,
    psize=10,
    interval=50,
    label=None
):
    """
    Creates an animation of vertical particle movement results.

    Parameters
    ----------
    lon : ndarray
        Longitude of particles (trajectory, time).
    depth : ndarray
        Depth of particles (trajectory, time).
    origin_marker : ndarray
        Origin marker of particles (trajectory, time).
    time_values : array-like
        Time values.
    psize : int, default=10
        Size of the scatter points.
    interval : int, default=50
        Delay between frames in milliseconds.
    label : list, optional
        Labels corresponding to each origin marker.
    """

    # Convert inputs to NumPy arrays
    lon = np.asarray(lon)
    depth = np.asarray(depth)
    origin_marker = np.asarray(origin_marker)

    # Validate array dimensions
    if lon.ndim != 2 or depth.ndim != 2 or origin_marker.ndim != 2:
        raise ValueError("lon, depth, and origin_marker must be 2D arrays.")

    if lon.shape != depth.shape or lon.shape != origin_marker.shape:
        raise ValueError(
            "lon, depth, and origin_marker must have the same shape."
        )

    if lon.shape[1] != len(time_values):
        raise ValueError(
            "The number of time values must match the second dimension "
            "of the particle arrays."
        )

    if interval <= 0:
        raise ValueError("interval must be greater than 0.")

    # Get unique origin markers for color mapping
    marker_values = np.unique(origin_marker)
    expected_values = np.arange(len(marker_values))

    if not np.array_equal(marker_values, expected_values):
        raise ValueError(
            "Origin markers should be consecutive integers starting from 0. "
            f"Found: {marker_values}, expected: {expected_values}"
        )

    # Set labels
    if label is None:
        label = [str(value) for value in marker_values]

    if len(label) != len(marker_values):
        raise ValueError(
            "The number of labels must match the number of unique "
            "origin markers. "
            f"Markers: {len(marker_values)}, labels: {len(label)}"
        )

    # Check valid particle positions
    valid = np.isfinite(lon) & np.isfinite(depth)

    if not np.any(valid):
        raise ValueError("No valid particle positions were found.")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    title = ax.set_title("Time: ")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Depth")

    # Set plot limits using all particle positions
    x_min, x_max = np.min(lon[valid]), np.max(lon[valid])
    y_min, y_max = np.min(depth[valid]), np.max(depth[valid])

    x_pad = (x_max - x_min) * 0.05 or 0.01
    y_pad = (y_max - y_min) * 0.05 or 0.01

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Background color
    if y_min < 0:
        ax.axhspan(
            y_min - y_pad,
            0,
            color="lightblue",
            alpha=0.3,
            zorder=0
        )

    # Create a colormap
    cmap = plt.get_cmap("viridis", len(label))

    scatter = ax.scatter(
        [], [],
        c=[],
        cmap=cmap,
        vmin=-0.5,
        vmax=len(label) - 0.5,
        s=psize,
        alpha=0.7
    )

    # Create legend
    handles = [
        Line2D(
            [0], [0], # Dummy data for the legend
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=7,
            label=name
        )
        for i, name in enumerate(label)
    ]

    ax.legend(
        handles=handles,
        title="Seeding Location",
        loc="lower left",
        )

    # Update function
    def update(frame):
        current_time = time_values[frame]

        # Update title
        title.set_text(f"Time: {current_time}")

        # Update particle positions
        positions = np.column_stack(
            (lon[:, frame], depth[:, frame])
        )
        scatter.set_offsets(positions)

        # Update particle colors
        scatter.set_array(origin_marker[:, frame])

        return scatter, title

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(time_values),
        interval=interval,
        blit=False
    )

    return ani