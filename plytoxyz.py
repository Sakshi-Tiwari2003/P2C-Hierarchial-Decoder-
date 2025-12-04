import os
import open3d as o3d

# Hardcoded input and output paths
ply_path = "/home/tejas/Downloads/Partial2Complete-main/experiments/P2C/EPN3D_models/test_epn3d_lamp/predictions/00011_ced76fc046191db3fe5c8ffd0f5eba47_pred.ply"
output_dir = "/home/tejas/Downloads/Partial2Complete-main/experiments/P2C/EPN3D_models/test_epn3d_lamp/predictions"

def ply_to_xyz(ply_path, output_dir=None):
    if not os.path.exists(ply_path):
        print(f"❌ Error: File {ply_path} does not exist.")
        return

    # Load the PLY file
    pcd = o3d.io.read_point_cloud(ply_path)
    points = pcd.points
    print(f"✅ Loaded {len(points)} points from: {ply_path}")

    # Set output .xyz file path
    filename = os.path.basename(ply_path).replace('.ply', '.xyz')
    xyz_path = os.path.join(output_dir, filename)

    # Save points to .xyz
    with open(xyz_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"✅ Saved XYZ file to: {xyz_path}")

if __name__ == "__main__":
    ply_to_xyz(ply_path, output_dir)
