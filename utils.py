import os
import json

TRAIN = "train"
VALID = "valid"
DEV = "test"
SPLITS = [TRAIN, VALID, DEV]

def load_dataset(args):
    context_data_path = os.path.join(args.data_dir, "context.json")
    with  open(context_data_path, 'r', encoding="utf-8") as f:
        context = json.load(f)
    
    data = {}
    for split in SPLITS:
        data_paths = os.path.join(args.data_dir, f"{split}.json")
        with  open(data_paths, 'r', encoding="utf-8") as f:
            data[split] = json.load(f)
    return context, data