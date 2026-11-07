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
        - time_values: pandas datetime index of time values
    """

    ds = xr.open_dataset(nc_path)

    lon = ds['lon'].to_numpy()
    lat = ds['lat'].to_numpy()
    depth = ds['z'].to_numpy()
    time_values = pd.to_datetime(ds['time'].to_numpy())

    ds.close()

    return lon, lat, depth, time_values

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
    nc_path,
    nodes,
    open_boundary_nodes,
    land_boundary_nodes,
    interval=100,
    save_path=None,
    stride=1,
):
    """
    OpenDriftのNetCDFから粒子の移動アニメーションを作成する。

    Parameters
    ----------
    nc_path : str
        OpenDriftの出力NetCDFファイル
    nodes : list
        [node_id, lon, lat, depth] 形式などのNodeデータ
    open_boundary_nodes : list
        開境界ごとのNode IDリスト
    land_boundary_nodes : list
        陸境界ごとのNode IDリスト
    interval : int
        アニメーションのフレーム間隔（ms）
    save_path : str or None
        保存先。例: "particles.mp4", "particles.gif"
    stride : int
        時間方向の間引き幅。1なら全時刻を使用
    """

    # NetCDF読み込み
    ds = xr.open_dataset(nc_path, decode_coords=False)

    try:
        # 座標を配列に変換
        lon = ds["lon"].transpose("time", "trajectory").to_numpy()
        lat = ds["lat"].transpose("time", "trajectory").to_numpy()

        # 想定する配列形状：(time, trajectory)
        # 逆の場合は転置
        if lon.ndim != 2 or lat.ndim != 2:
            raise ValueError(
                f"lon/latは2次元を想定しています: "
                f"lon={lon.shape}, lat={lat.shape}"
            )

        if lon.shape != lat.shape:
            raise ValueError("lonとlatの配列形状が一致しません。")

        if ds["time"].dims == ("time",):
            time_values = pd.to_datetime(ds["time"].to_numpy())
        else:
            raise ValueError("time変数の次元が想定と異なります。")

        # 時間が第0軸でない場合は転置
        if ds["lon"].dims != ("time", "trajectory"):
            if set(ds["lon"].dims) == {"time", "trajectory"}:
                lon = ds["lon"].transpose("time", "trajectory").to_numpy()
                lat = ds["lat"].transpose("time", "trajectory").to_numpy()
            else:
                raise ValueError(
                    f"想定外のlonの次元: {ds['lon'].dims}"
                )

        # statusがあれば、無効な粒子位置を除外
        if "status" in ds:
            status = ds["status"].transpose(
                "time", "trajectory"
            ).values
            valid_status = status >= 0
        else:
            valid_status = np.ones(lon.shape, dtype=bool)

        # 描画に使う時刻を間引く
        frame_indices = np.arange(0, len(time_values), stride)

        # 海岸線を描画
        fig, ax = draw_coastline(
            nodes,
            open_boundary_nodes,
            land_boundary_nodes,
        )

        # 全時刻の粒子位置を1つのScatterで管理
        scat = ax.scatter(
            [], [],
            s=12,
            color="red",
            zorder=5,
            label="Particles",
        )

        title = ax.set_title("")
        ax.legend()

        # 描画範囲を固定
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        def update(frame):
            t = frame_indices[frame]

            # 有効な粒子かつ座標が有限のものだけ表示
            valid = (
                valid_status[t]
                & np.isfinite(lon[t])
                & np.isfinite(lat[t])
            )

            points = np.column_stack(
                (lon[t, valid], lat[t, valid])
            )

            scat.set_offsets(points)
            title.set_text(
                f"Time: {time_values[t]:%Y-%m-%d %H:%M}"
            )

            return scat, title

        ani = FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=interval,
            blit=False,
        )

        if save_path is not None:
            if save_path.endswith(".gif"):
                ani.save(save_path, writer="pillow", fps=10)
            elif save_path.endswith(".mp4"):
                ani.save(save_path, writer="ffmpeg", fps=10)
            else:
                raise ValueError(
                    "保存形式は .gif または .mp4 にしてください。"
                )

        plt.show()

        return ani

    finally:
        ds.close()