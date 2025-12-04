import os
import torch
import numpy as np
import open3d as o3d

def point_display(points):
    '''
    input: (1, num_point, 3) or (num_points, 3)
    '''
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    points = np.squeeze(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def point_save(points, path, file_name, type='ply'):
    '''
    Saves a point cloud as .ply or .pcd
    Inputs:
        - points: torch tensor or numpy array of shape (1, N, 3) or (N, 3)
        - path: folder to save in
        - file_name: name without extension
        - type: 'ply' (default) or 'pcd'
    '''
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()

    # Ensure shape is (N, 3)
    points = np.squeeze(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid point shape: expected (N, 3), got {points.shape}")

    # Ensure correct dtype
    points = points.astype(np.float32)

    # Create and save point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path)

    save_path = os.path.join(path, f"{file_name}.{type}")
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)



def to_point_cloud(points):
    '''
    input: (1, num_point, 3) or (num_points, 3)
    '''
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    points = np.squeeze(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def to_point_cloud_with_color(points, colors):
    '''
    input: (1, num_point, 3) or (num_points, 3)
    '''
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().detach().numpy()
    points = np.squeeze(points)
    colors = np.squeeze(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors= o3d.utility.Vector3dVector(colors)
    return pcd


def o3d_point_save(points, path, file_name, type='ply'):
    o3d.io.write_point_cloud(os.path.join(path, file_name+'.'+type), points, write_ascii=True)