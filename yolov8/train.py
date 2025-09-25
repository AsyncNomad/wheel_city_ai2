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
    ap.add_argument("--model",    default=os.environ.get("YOLO_MODEL", "yolov8m.pt"))
    ap.add_argument("--data",     default=os.environ.get("YOLO_DATA", "yolov8/data.yaml"))
    ap.add_argument("--epochs",   type=int, default=int(os.environ.get("EPOCHS", "50")))
    ap.add_argument("--imgsz",    type=int, default=int(os.environ.get("IMGSZ", "640")))
    ap.add_argument("--batch",    type=int, default=int(os.environ.get("BATCH", "8")))
    ap.add_argument("--name",     default=os.environ.get("RUN_NAME", "ver1"))
    ap.add_argument("--patience", type=int, default=int(os.environ.get("PATIENCE", "10")))
    args = ap.parse_args()

    print("Starting training with the following configuration:")
    print(f"- Model: {args.model}")
    print(f"- Data: {args.data}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Image Size: {args.imgsz}")
    print(f"- Batch Size: {args.batch}")
    print(f"- Patience: {args.patience}")
    print(f"- Run Name: {args.name}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="yolov8/train_result",  # 결과물 저장 경로
        name=args.name,
        patience=args.patience,
        pretrained=True
    )
    print("Training finished.")
    print(f"Results saved to: yolov8/train_result/{args.name}")

if __name__ == "__main__":
    main()