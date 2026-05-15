# Visualize all the normalized microlane images by putting ground truths on top of the image
# Would want to do this recursively as well

import argparse, cv2, json
from pathlib import Path
import numpy as np

from microlane.utils.load_config import load_config

config = load_config()

COLORS = config.constants.colors.lane_colors

def draw_ground_truth(img: np.ndarray, lanes: list[list[int]], h_samples: list[int]) -> np.ndarray:
    out = img.copy()
 
    for i, lane in enumerate(lanes):
        color = COLORS[i % len(COLORS)]
        pts = [(x, h) for x, h in zip(lane, h_samples) if x != -2]
        for j in range(len(pts) - 1):
            cv2.line(out, pts[j], pts[j + 1], color=color, thickness=2)
        for x, y in pts:
            cv2.circle(out, (x, y), radius=3, color=color, thickness=-1)
 
    return out

def main() -> None:
    
    p = argparse.ArgumentParser()
 
    p.add_argument("--images",       required=True)  # Path to normalized images root (clip_N/...)
    p.add_argument("--annotations",  required=True)  # Path to annotations.json
    p.add_argument("--output",       default="results/visualized_microlane")
 
    args = p.parse_args()
 
    img_root = Path(args.images)
    ann_path = Path(args.annotations)
    out_root = Path(args.output)
 
    print(f"[*] Loading annotations from {ann_path}")
    entries: list[dict] = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
 
    print(f"    {len(entries)} entries found")
 
    for entry in entries:
        raw_file  = entry["raw_file"]    # e.g. "clip_3/frame_0001.jpg"
        lanes     = entry["lanes"]
        h_samples = entry["h_samples"]
 
        img_path = img_root / raw_file
        if not img_path.exists():
            print(f"  [!] Missing image {img_path}, skipping")
            continue
 
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [!] Could not read {img_path}, skipping")
            continue
 
        out_path = out_root / raw_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
 
        vis = draw_ground_truth(img, lanes, h_samples)
        cv2.imwrite(str(out_path), vis)
 
    print(f"\n[+] Done → {out_root}")

if __name__ == "__main__":
    main()