import os
import open3d as o3d
import numpy as np

# Hardcoded input and output paths
xyz_path = "/home/tejas/Downloads/Partial2Complete-main/experiments/P2C/EPN3D_models/test_epn3d_lamp/predictions/densetree2.xyz"
output_dir = "/home/tejas/Downloads/Partial2Complete-main/experiments/P2C/EPN3D_models/test_epn3d_lamp/predictions"

def xyz_to_ply(xyz_path, output_dir=None):
    if not os.path.exists(xyz_path):
        print(f"❌ Error: File {xyz_path} does not exist.")
        return

    # Load points from .xyz file
    points = []
    with open(xyz_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    if len(points) == 0:
        print("⚠️ No points loaded from file.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # Set output path
    filename = os.path.basename(xyz_path).replace('.xyz', '_reconverted.ply')
    ply_path = os.path.join(output_dir, filename)

    # Write PLY file
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"✅ Saved PLY file to: {ply_path}")

if __name__ == "__main__":
    xyz_to_ply(xyz_path, output_dir)
