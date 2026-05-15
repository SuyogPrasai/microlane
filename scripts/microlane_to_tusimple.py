# Convert the microlane dataset to TuSimple format
# Rescale from 1920x1080 to 1280x720
# Convert polylines to h_samples format
# We would want to do this recursively because in the microlane folder there are specific clip folders, and we would like to emulate that in the parsed images as well


import argparse, json, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2

from microlane.utils.load_config import load_config

config = load_config()

SRC_H, SRC_W = config.datasets.microlane.dimensions
FINAL_H, FINAL_W = config.datasets.tusimple.dimensions
H_SAMPLES = config.constants.h_samples

def build_filename_to_clip(src_dir: Path) -> dict[str, str]:
    mapping = {}
    for clip_dir in sorted(src_dir.iterdir()):
        if clip_dir.is_dir() and clip_dir.name.startswith("clip_"):
            for f in clip_dir.iterdir():
                if f.is_file():
                    mapping[f.name] = clip_dir.name
    return mapping

def resize_images(src_dir: Path, dist_dir: Path, target_w: int, target_h: int):
    for clip_dir in sorted(src_dir.iterdir()):
        if not (clip_dir.is_dir() and clip_dir.name.startswith("clip_")):
            continue
 
        out_clip = dist_dir / clip_dir.name
        out_clip.mkdir(parents=True, exist_ok=True)
 
        frames = sorted(f for f in clip_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for frame in frames:
            img = cv2.imread(str(frame))
            if img is None:
                print(f"  [!] Could not read {frame}, skipping")
                continue
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_clip / frame.name), resized)
 
        print(f"  [+] {clip_dir.name}: {len(frames)} frames → {out_clip}")
        

def parse_cvat_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
 
    for image_elem in root.iter("image"):
        image_name = Path(image_elem.get("name", "")).name  # bare filename only
 
        lanes = []
        for poly in image_elem.iter("polyline"):
            raw = poly.get("points", "")
            if not raw:
                continue
            pts = []
            for pair in raw.split(";"):
                x_str, y_str = pair.strip().split(",")
                pts.append((float(x_str), float(y_str)))
            if pts:
                lanes.append(pts)
 
        records.append({"image_name": image_name, "lanes": lanes})
 
    return records

def interpolate_x(points: list[tuple], y_query: float) -> float:
    pts = sorted(points, key=lambda p: p[1])
 
    if y_query < pts[0][1] or y_query > pts[-1][1]:
        return -2
 
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if y0 <= y_query <= y1:
            if y1 == y0:
                return x0
            t = (y_query - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return -2


def to_tusimple(
    records: list[dict],
    filename_to_clip: dict[str, str],
    h_samples: list[int],
    sx: float,
    sy: float,
) -> list[dict]:
    entries = []
 
    for rec in records:
        name = rec["image_name"]
        clip = filename_to_clip.get(name)
        if clip is None:
            print(f"  [!] No clip found for '{name}', skipping")
            continue
 
        lanes_out = []
        for lane_pts in rec["lanes"]:
            scaled = [(x * sx, y * sy) for x, y in lane_pts]
            xs = [round(interpolate_x(scaled, h)) for h in h_samples]
            lanes_out.append(xs)
 
        entries.append({
            "raw_file":  f"{clip}/{name}",
            "lanes":     lanes_out,
            "h_samples": h_samples,
        })
 
    return entries


def main() -> None:
    
    p = argparse.ArgumentParser()
    
    p.add_argument("--annotations",   required=True) # Path to XML annotation files
    p.add_argument("--microlane",     required=True) # Path to Microlane UnModified Files
    p.add_argument("--modified",      required=True) # Path to Microlane Modfied Files
    
    p.add_argument("--output",        default="results/normalized_microlane")
    
    p.add_argument("--target-width",  type=int, default=FINAL_W)
    p.add_argument("--target-height", type=int, default=FINAL_H)
    
    args = p.parse_args()

    ann_path = Path(args.annotations)
    ml_dir   = Path(args.microlane)
    mod_dir  = Path(args.modified)
    
    out_root = Path(args.output)
    tw, th   = args.target_width, args.target_height
    sx, sy   = tw / SRC_W, th / SRC_H
    
    print(f"[*] Resizing the images to [{FINAL_W}x{FINAL_H}]")
    print(f"[*] Parsing the XML CVAT Image Annotation to TuSimple format with fixed h_samples")
    
    ml_out  = out_root / "microlane"
    mod_out = out_root / "modified_microlane"
 
    print("\n[*] Resizing Microlane...")
    resize_images(ml_dir, ml_out, tw, th)
 
    print("\n[*] Resizing Modified Microlane...")
    resize_images(mod_dir, mod_out, tw, th)
    
    filename_to_clip = build_filename_to_clip(ml_dir)
    print(f"\n[*] Found {len(filename_to_clip)} frames across clips")
 
    print(f"\n[*] Parsing {ann_path}")
    records = parse_cvat_xml(ann_path)
    print(f"    {len(records)} annotated images")
 
    print("\n[*] Converting to TuSimple...")
    tusimple_data = to_tusimple(records, filename_to_clip, H_SAMPLES, sx, sy)
    print(f"    {len(tusimple_data)} entries generated")
 
    out_root.mkdir(parents=True, exist_ok=True)
    ann_out = out_root / "annotations.json"
    with open(ann_out, "w") as f:
        for entry in tusimple_data:
            f.write(json.dumps(entry) + "\n")
 
    print(f"\n[+] Done → {ann_out}")
    
if __name__ == "__main__":
    main()    
