import json
import random
import os

# === Configuration ===
input_path = 'data/EPN3D/EPN3D.json'         # original JSON
output_path = 'data/EPN3D/EPN3D_split.json'  # new JSON with val split
val_ratio = 0.1                              # 10% of train goes to val
random_seed = 42

# === Load original JSON ===
with open(input_path, 'r') as f:
    data = json.load(f)

# === Apply split to each class ===
random.seed(random_seed)
for category in data:
    train_partial = category['train']['partial']
    train_complete = category['train']['complete']
    assert len(train_partial) == len(train_complete), "Mismatch in partial/complete length"

    indices = list(range(len(train_partial)))
    random.shuffle(indices)

    split_idx = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    # Split data
    category['train']['partial'] = [train_partial[i] for i in train_idx]
    category['train']['complete'] = [train_complete[i] for i in train_idx]

    category['val'] = {
        'partial': [train_partial[i] for i in val_idx],
        'complete': [train_complete[i] for i in val_idx]
    }

print(f"✅ Split complete. Saving to: {output_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print("🎉 Done.")
