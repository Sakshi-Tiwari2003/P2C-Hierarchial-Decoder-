import os
import open3d as o3d
import numpy as np

base_path = "data/PCN"  # Adjust if your path is different
splits = ["train", "val", "test"]
folders = ["partial", "complete"]

for split in splits:
    for folder in folders:
        dir_path = os.path.join(base_path, split, folder)
        if not os.path.exists(dir_path):
            continue

        for synset_id in os.listdir(dir_path):
            synset_path = os.path.join(dir_path, synset_id)
            if not os.path.isdir(synset_path):
                continue

            if folder == "complete":
                # Format: data/PCN/train/complete/03001627/file.pcd
                for file in os.listdir(synset_path):
                    if file.endswith(".pcd"):
                        file_path = os.path.join(synset_path, file)
                        pcd = o3d.io.read_point_cloud(file_path)
                        points = np.asarray(pcd.points)
                        np.save(file_path.replace(".pcd", ".npy"), points)

            elif folder == "partial":
                # Format: data/PCN/train/partial/03001627/<model_id>/00.pcd
                for model_id in os.listdir(synset_path):
                    model_dir = os.path.join(synset_path, model_id)
                    if not os.path.isdir(model_dir):
                        continue
                    for file in os.listdir(model_dir):
                        if file.endswith(".pcd"):
                            file_path = os.path.join(model_dir, file)
                            pcd = o3d.io.read_point_cloud(file_path)
                            points = np.asarray(pcd.points)
                            np.save(file_path.replace(".pcd", ".npy"), points)

print("✅ All PCD files in both 'complete' and 'partial' folders converted to .npy")
