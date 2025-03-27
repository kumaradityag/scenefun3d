import open3d as o3d
import argparse
from pathlib import Path
from utils.data_parser import DataParser
from utils.pc_process import pc_estimate_normals
from utils.viz import viz_3d


def main(args):
    data_parser = DataParser(args.data_dir)
    laser_scan = data_parser.get_laser_scan(args.visit_id)
    laser_scan = data_parser.get_cropped_laser_scan(args.visit_id, laser_scan)
    pcd = laser_scan
    pcd.voxel_down_sample(voxel_size=0.05)
    # map_path = Path(args.map_path)
    # pcd = o3d.io.read_point_cloud(str(map_path / "point_cloud.pcd"))
    pcd = pc_estimate_normals(pcd)

    viz_3d([pcd], viz_tool="open3d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Path of the data")
    parser.add_argument("--visit_id", required=True, help="Identifier of the scene")
    # parser.add_argument(
    #     "--map_path", required=True, help="Path of the concept-nodes map"
    # )
    args = parser.parse_args()

    main(args)
