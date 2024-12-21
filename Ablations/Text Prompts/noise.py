import os
import subprocess

# Create the output directory
os.makedirs("/kaggle/working/sample4/", exist_ok=True)

# Change directory
%cd /kaggle/working/InstagenDemo/mmdetection

# Initialize image counter
i = 1

# Define categories
categories = [
    "cup, sandwich",
    "umbrella, bench",
    "cat, carrot",
    "bear, cake",
    "dog, boat",
    "elephant, boat",
    "vase, skateboard",
    "cow, banana",
    "person, cake",
    "umbrella, motorcycle"
]

# Process each category
for cat in categories:
    print(f"\nON IMAGE NUMBER: {i}\n")
    i += 1

    # Format class names and output file
    class_names = cat.replace(", ", "_")
    output_file = f"/kaggle/working/sample4/{class_names}.jpg"

    # Build and run the command
    command = [
        "python", "demo.py",
        "--detector_config", "configs/instagen/instagen-4scale_fd_8xb2-12e_coco_demo.py",
        "--detector_ckpt", "/kaggle/input/instagen-model-weights/pytorch/default/1/instagen-4scale_fd_8xb2-12e_coco.pth",
        "--cls_names", cat,
        "--score_thr", "0.4",
        "--out", output_file
    ]
    subprocess.run(command, check=True)
