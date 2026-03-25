import os

label_dirs = ['train/labels', 'valid/labels', 'test/labels']
base_path = r'D:\DMS\Driver-Monitoring-System-main/dataset'

invalid_files = []

for folder in label_dirs:
    full_path = os.path.join(base_path, folder)
    if not os.path.exists(full_path):
        print(f"⚠ Folder not found: {full_path}")
        continue

    for file in os.listdir(full_path):
        if file.endswith('.txt'):
            with open(os.path.join(full_path, file), 'r') as f:
                for line in f:
                    cls = int(line.split()[0])
                    if cls not in [0, 1]:  # Only 'belt' and 'phone' allowed now
                        invalid_files.append((file, cls))

if invalid_files:
    print("\n❌ Found invalid labels:")
    for file, cls in invalid_files:
        print(f"   File: {file} -> Contains class: {cls}")
else:
    print("\n✅ All labels are clean (only class 0 and 1 present).")
