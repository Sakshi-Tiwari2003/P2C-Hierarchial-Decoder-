import os

base_path = "data/PCN"  # Adjust this path if needed
deleted_files = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".pcd"):
            full_path = os.path.join(root, file)
            try:
                os.remove(full_path)
                deleted_files.append(full_path)
            except Exception as e:
                print(f"Failed to delete {full_path}: {e}")

print(f"✅ Deleted {len(deleted_files)} .pcd files.")
