# fix_clean_pcn_test_split.py
import json, os
from tqdm import tqdm

json_path = "data/PCN/PCN.json"
category_id = "04379243"
n_test = 555
partial_base = f"data/PCN/test/partial/{category_id}"
complete_base = f"data/PCN/test/complete/{category_id}"

# Load existing JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Find 'table' category
table_entry = next((c for c in data if c["taxonomy_id"] == category_id), None)
used_ids = set(table_entry["train"] + table_entry["val"])

# Scan valid test IDs
valid_ids = []
for model_id in tqdm(os.listdir(partial_base)):
    if model_id in used_ids:
        continue
    part_path = os.path.join(partial_base, model_id, "00.npy")
    comp_path = os.path.join(complete_base, f"{model_id}.npy")
    if os.path.exists(part_path) and os.path.exists(comp_path):
        valid_ids.append(model_id)
    if len(valid_ids) >= n_test:
        break

# Update and save
table_entry["test"] = valid_ids
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"✅ Added {len(valid_ids)} test samples for 'table' with verified files.")
