import os

# --- Configuration ---
dataset_dir = r"D:\DMS\Driver-Monitoring-System-main\dataset"
labels_dir = os.path.join(dataset_dir, "labels")
output_dir = os.path.join(dataset_dir, "labels_cleaned")
iou_threshold = 0.85  # consider boxes duplicates if IoU > this

os.makedirs(output_dir, exist_ok=True)

def iou(box1, box2):
    """Compute IoU between two YOLO-format boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert from center to corners
    x1_min, y1_min, x1_max, y1_max = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
    x2_min, y2_min, x2_max, y2_max = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area != 0 else 0

def remove_duplicates_from_file(filepath, output_path):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    boxes = []
    cleaned_lines = []

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls, x, y, w, h = parts
        x, y, w, h = map(float, (x, y, w, h))
        current_box = (x, y, w, h)

        # Check IoU against existing boxes
        duplicate = False
        for existing_box in boxes:
            if iou(current_box, existing_box) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            boxes.append(current_box)
            cleaned_lines.append(line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(cleaned_lines))

# --- Main loop ---
total_files = 0
duplicates_removed = 0

# Walk through train, val, test folders
for root, _, files in os.walk(labels_dir):
    for file in files:
        if file.endswith(".txt"):
            total_files += 1
            input_path = os.path.join(root, file)

            # Mirror same subfolder in output_dir
            relative_path = os.path.relpath(input_path, labels_dir)
            output_path = os.path.join(output_dir, relative_path)

            with open(input_path, "r") as f:
                original_lines = len(f.readlines())

            remove_duplicates_from_file(input_path, output_path)

            with open(output_path, "r") as f:
                new_lines = len(f.readlines())

            if new_lines < original_lines:
                duplicates_removed += 1
                print(f"Cleaned {relative_path}: {original_lines - new_lines} duplicates removed")

print(f"\n✅ Done! Processed {total_files} label files, cleaned {duplicates_removed}.")
print(f"Cleaned labels saved to: {output_dir}")

