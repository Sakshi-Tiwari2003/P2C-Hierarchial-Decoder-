import os
import json

json_path = "data/PCN/PCN.json"

# Load the existing PCN.json
with open(json_path, "r") as f:
    data = json.load(f)

# Check if all partials and gt file exist
def file_exists(category, model_id, subset, num_partial=8):
    base_dir = f"data/PCN/{subset}/partial/{category}/{model_id}"
    gt_path = f"data/PCN/{subset}/complete/{category}/{model_id}.npy"
    if not os.path.exists(gt_path):
        return False
    for i in range(num_partial):
        part_path = os.path.join(base_dir, f"{i:02d}.npy")
        if not os.path.exists(part_path):
            return False
    return True

# Filter the dataset
filtered_data = []
for entry in data:
    if entry["taxonomy_id"] != "04379243":
        filtered_data.append(entry)
        continue

    new_entry = {
        "taxonomy_id": entry["taxonomy_id"],
        "taxonomy_name": entry["taxonomy_name"],
        "train": [],
        "val": [],
        "test": []
    }

    for subset in ["train", "val", "test"]:
        for model_id in entry.get(subset, []):
            if file_exists(entry["taxonomy_id"], model_id, subset):
                new_entry[subset].append(model_id)

    filtered_data.append(new_entry)

# Overwrite the existing PCN.json
with open(json_path, "w") as f:
    json.dump(filtered_data, f, indent=2)

print(" PCN.json successfully filtered and overwritten.")
