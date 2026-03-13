import os

# Paths
CHECKPOINTS_DIR = os.path.join('dashboard', 'backend', 'results_ms', '1_Training', 'checkpoints')
MAPPING_FILE = os.path.join('plan', 'mapping.txt')

def main():
    # 1. Ensure Plan directory exists
    os.makedirs('plan', exist_ok=True)

    # 2. Get current task list from checkpoints
    if not os.path.exists(CHECKPOINTS_DIR):
        print(f"Error: Checkpoints directory not found: {CHECKPOINTS_DIR}")
        return

    current_tasks = [d for d in os.listdir(CHECKPOINTS_DIR) if os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))]
    
    # 3. Load existing mapping if it exists
    existing_mapping = {}
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        existing_mapping[parts[0].strip()] = parts[1].strip()

    # 4. Update mapping file with current tasks
    print(f"Updating {MAPPING_FILE}...")
    with open(MAPPING_FILE, 'w') as f:
        for task in sorted(current_tasks):
            path = existing_mapping.get(task, "FILL_ME_MANUALLY")
            f.write(f"{task} | {path}\n")

    # 5. Process entries that have a valid path
    print("\nProcessing task data folders...")
    for task, run_data_path in existing_mapping.items():
        if not run_data_path or run_data_path == "FILL_ME_MANUALLY":
            continue
        
        if not os.path.exists(run_data_path):
            print(f"  ⚠️  Path not found for {task}: {run_data_path}")
            continue

        # Target: run_data/train_ag
        train_ag_path = os.path.join(run_data_path, 'train_ag')
        if not os.path.exists(train_ag_path):
            # Try lowercase variations or common patterns
            found = False
            for sub in ['train_ag', 'Train_AG', 'TRAIN_AG']:
                if os.path.exists(os.path.join(run_data_path, sub)):
                    train_ag_path = os.path.join(run_data_path, sub)
                    found = True
                    break
            if not found:
                print(f"  ⚠️  'train_ag' folder not found in {run_data_path}")
                continue

        # Extract sample IDs from filenames
        sample_ids = []
        for filename in os.listdir(train_ag_path):
            if os.path.isfile(os.path.join(train_ag_path, filename)):
                # Strip extension (e.g. Sample_1.txt -> Sample_1)
                sample_id = os.path.splitext(filename)[0]
                sample_ids.append(sample_id)

        if sample_ids:
            target_file = os.path.join(CHECKPOINTS_DIR, task, "train_samples.txt")
            with open(target_file, 'w') as f_out:
                f_out.write("\n".join(sorted(sample_ids)))
            print(f"  ✅ Generated train_samples.txt for {task} ({len(sample_ids)} samples)")
        else:
            print(f"  ⚠️  No samples found in {train_ag_path}")

    print("\nDone. Please fill in/verify paths in plan/mapping.txt and run this script again if needed.")

if __name__ == "__main__":
    main()
