#!/usr/bin/env python3
# yolov8/train.py
import os
import argparse
from ultralytics import YOLO
from dotenv import load_dotenv

# .env 파일을 환경변수로 로드
load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default=os.environ.get("YOLO_MODEL", "yolov8m.pt"))
    ap.add_argument("--data",   default=os.environ.get("YOLO_DATA", "yolov8/data_balanced.yaml"))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "50")))
    ap.add_argument("--imgsz",  type=int, default=int(os.environ.get("IMGSZ", "640")))
    ap.add_argument("--batch",  type=int, default=int(os.environ.get("BATCH", "8")))
    ap.add_argument("--name",   default=os.environ.get("RUN_NAME", "ver1"))
    args = ap.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="yolov8/train_result",  # 결과물 저장 경로
        name=args.name,
        pretrained=True
    )
    print(results)

if __name__ == "__main__":
    main()
