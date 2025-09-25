#!/usr/bin/env python3
# yolov8/run.py
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8/train_result/ver1/weights/best.pt")
    ap.add_argument("--source",  default="input_images")
    ap.add_argument("--outdir",  default="bbox_images")
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--conf",    type=float, default=0.25)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=args.outdir,
        name=".",
        exist_ok=True,
        line_width=2
    )
    print(f"[OK] Saved bbox images to: {Path(args.outdir).resolve()}")

if __name__ == "__main__":
    main()
