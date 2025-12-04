import json
import random

# Path to the existing filtered PCN.json
json_path = "data/PCN/PCN.json"

# Load the current data
with open(json_path, 'r') as f:
    data = json.load(f)

# Process only the 'table' class
for cat in data:
    if cat["taxonomy_name"] == "table":
        train_list = cat.get("train", [])
        
        if len(train_list) < 100:
            raise ValueError("Not enough training samples to reassign to validation.")

        # Sample 100 for val set
        val_samples = random.sample(train_list, 100)

        # Update val list
        cat["val"] = val_samples

        # Remove those from train
        cat["train"] = [mid for mid in train_list if mid not in val_samples]

        # Clear test set (optional, or keep as is if known good IDs)
        cat["test"] = []

        print(f"Moved {len(val_samples)} samples to validation for class 'table'")
        break

# Save updated JSON
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print("PCN.json updated and saved in-place.")
