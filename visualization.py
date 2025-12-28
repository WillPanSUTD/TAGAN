import open3d as o3d
import numpy as np


def save_visualization(colored_data, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(colored_data[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(colored_data[:, 3:6] / 255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_path))
    vis.destroy_window()