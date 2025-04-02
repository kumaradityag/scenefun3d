import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import random

from utils.data_parser import DataParser
from utils.pc_process import pc_estimate_normals


def generate_random_colors(n):
    """
    Generates `n` distinct RGB colors in [0, 1]
    """
    random.seed(42)
    colors = []
    for _ in range(n):
        color = [random.random(), random.random(), random.random()]
        colors.append(color)
    return np.array(colors)


def main(args):
    map_path = Path(args.map_path)

    # Load laser scan
    data_parser = DataParser(args.data_dir)
    laser_scan = data_parser.get_laser_scan(args.visit_id)
    laser_pts = np.asarray(laser_scan.points)
    num_laser_pts = laser_pts.shape[0]

    # Load functional masks
    masks = np.load(map_path / "functional_masks_laser_scan.npy")
    num_func_instances = masks.shape[0]

    print(f"Laser scan points: {num_laser_pts}")
    print(f"Functional instances: {num_func_instances}")

    # Start with gray color for all
    colors = np.ones((num_laser_pts, 3)) * 0.6  # light gray

    # Generate random color per functionality instance
    func_colors = generate_random_colors(num_func_instances)

    # Color each point based on mask membership
    for i in range(num_func_instances):
        mask = masks[i]
        colors[mask] = func_colors[i]  # overwrite

    # Assign and show
    laser_scan.colors = o3d.utility.Vector3dVector(colors)
    laser_scan = data_parser.get_cropped_laser_scan(args.visit_id, laser_scan)
    laser_scan = laser_scan.voxel_down_sample(voxel_size=0.025)
    # laser_scan = pc_estimate_normals(laser_scan)
    # o3d.visualization.draw_geometries([laser_scan], window_name="Functional Masks on Laser Scan")
    o3d.io.write_point_cloud(str(map_path / "functional_masks_laser_scan.pcd"), laser_scan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Path of the data")
    parser.add_argument("--visit_id", required=True, help="Identifier of the scene")
    parser.add_argument("--map_path", required=True, help="Path of the concept-nodes map")
    args = parser.parse_args()
    main(args)
