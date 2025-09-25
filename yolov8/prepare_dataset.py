#!/usr/bin/env python3
# yolov8/build_balanced_dataset.py
import argparse
import random
from pathlib import Path
from lxml import etree
from tqdm import tqdm
import yaml

# 지원하는 이미지 확장자
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def parse_args():
    """커맨드라인 인자 파싱"""
    ap = argparse.ArgumentParser(
        description="Parse XML, balance dataset (ramp:barrier:negative=1:1:1), and create YOLOv8 files."
    )
    ap.add_argument("--images_dir", default="yolov8/images", help="Path to the directory containing all images.")
    ap.add_argument("--xml_glob", default="yolov8/**/*.xml", help="Glob pattern to find all XML annotation files.")
    ap.add_argument("--output_dir", default="yolov8", help="Directory to save the final labels, splits, and yaml file.")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of the dataset to be used for validation.")
    return ap.parse_args()

def parse_xml_files(xml_paths, images_root):
    """
    모든 XML 파일을 파싱하여 이미지별 어노테이션 정보와 클래스 유형을 반환합니다.
    """
    image_data = {}  # {img_path: {"lines": [...], "classes": {...}}}
    image_path_map = {p.name: p for p in images_root.rglob("*") if p.suffix.lower() in IMG_EXTS}

    for xml_path in tqdm(xml_paths, desc="Parsing XML Annotations"):
        try:
            tree = etree.parse(str(xml_path))
        except etree.XMLSyntaxError:
            print(f"Warning: Could not parse XML file: {xml_path}")
            continue
            
        for im_elem in tree.findall(".//image"):
            name = im_elem.get("name")
            if not name:
                continue

            img_path = image_path_map.get(Path(name).name)
            if not img_path:
                continue
            
            try:
                W = float(im_elem.get("width", 0))
                H = float(im_elem.get("height", 0))
            except (ValueError, TypeError):
                continue
            
            if W <= 0 or H <= 0:
                continue

            lines = []
            classes_in_image = set()
            for box in im_elem.findall(".//box"):
                label = (box.get("label") or "").strip().lower()
                if label not in ("step", "stair", "ramp"):
                    continue
                
                try:
                    xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
                    xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
                except (ValueError, TypeError):
                    continue

                if not (xbr > xtl and ybr > ytl):
                    continue
                
                cx, cy = ((xtl + xbr) / 2.0) / W, ((ytl + ybr) / 2.0) / H
                bw, bh = (xbr - xtl) / W, (ybr - ytl) / H

                if not (0 < cx < 1 and 0 < cy < 1 and 0 < bw < 1 and 0 < bh < 1):
                    continue

                cls_id = 0 if label == "ramp" else 1  # 0: ramp, 1: barrier (step+stair)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                classes_in_image.add(cls_id)
            
            if lines:
                if img_path in image_data: # 이미 다른 XML 파일에서 파싱된 경우 라인 추가
                    image_data[img_path]["lines"].extend(lines)
                    image_data[img_path]["classes"].update(classes_in_image)
                else:
                    image_data[img_path] = {"lines": lines, "classes": classes_in_image}
                
    return image_data

def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "labels"

    assert images_dir.exists(), f"Image directory not found: {images_dir}"

    # 1. 모든 이미지 목록 확보 및 XML 파싱
    all_image_paths = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    xml_files = sorted(Path(".").glob(args.xml_glob))
    image_annotations = parse_xml_files(xml_files, images_dir)
    
    # 2. 이미지를 세 그룹으로 분류
    ramp_images = []
    barrier_only_images = []
    
    labeled_paths = set(image_annotations.keys())
    
    for img_path, data in image_annotations.items():
        if 0 in data["classes"]:
            ramp_images.append(img_path)
        elif 1 in data["classes"]:
            barrier_only_images.append(img_path)
            
    # 라벨링된 이미지를 제외한 나머지를 negative 이미지로 분류
    negative_images = [p for p in all_image_paths if p not in labeled_paths]

    print(f"Found {len(ramp_images)} images with ramps.")
    print(f"Found {len(barrier_only_images)} images with only barriers.")
    print(f"Found {len(negative_images)} negative images (no relevant labels).")

    # 3. 데이터셋 밸런싱 (1:1:1 비율)
    random.shuffle(barrier_only_images)
    random.shuffle(negative_images)
    
    # ramp 이미지 수를 기준으로 샘플링
    n_base = len(ramp_images)
    
    if n_base == 0:
        print("Error: No ramp images found. Cannot balance dataset.")
        return

    n_barriers_to_keep = min(len(barrier_only_images), n_base)
    n_negatives_to_keep = min(len(negative_images), n_base)
    
    selected_barriers = barrier_only_images[:n_barriers_to_keep]
    selected_negatives = negative_images[:n_negatives_to_keep]
    
    final_image_paths = ramp_images + selected_barriers + selected_negatives
    random.shuffle(final_image_paths)

    print("-" * 30)
    print(f"Balancing dataset to 1:1:1 ratio based on {n_base} ramp images...")
    print(f"Selected {len(ramp_images)} ramp images.")
    print(f"Selected {len(selected_barriers)} barrier images.")
    print(f"Selected {len(selected_negatives)} negative images.")
    print(f"Total images in balanced dataset: {len(final_image_paths)}")
    print("-" * 30)

    # 4. 학습 / 검증 데이터 분할
    n_total = len(final_image_paths)
    n_val = int(n_total * args.val_ratio)
    
    val_paths = final_image_paths[:n_val]
    train_paths = final_image_paths[n_val:]

    # 5. 라벨 파일 및 데이터셋 관련 파일 생성
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(final_image_paths, desc="Writing Label Files"):
        label_file = labels_dir / f"{img_path.stem}.txt"
        if img_path in image_annotations:
            # 라벨링된 이미지의 경우, 어노테이션 정보를 파일에 쓴다
            yolo_lines = image_annotations[img_path]["lines"]
            label_file.write_text("".join(yolo_lines), encoding="utf-8")
        else:
            # Negative 이미지의 경우, 빈 파일을 생성한다
            label_file.write_text("", encoding="utf-8")

    # train.txt, val.txt 생성
    train_txt_path = output_dir / "train.txt"
    val_txt_path = output_dir / "val.txt"
    train_txt_path.write_text("\n".join(str(p.resolve()) for p in train_paths) + "\n", encoding="utf-8")
    val_txt_path.write_text("\n".join(str(p.resolve()) for p in val_paths) + "\n", encoding="utf-8")
    
    # data.yaml 생성
    data_yaml_path = output_dir / "data.yaml"
    data = {
        "names": {0: "ramp", 1: "barrier"},
        "train": str(train_txt_path.resolve()),
        "val":   str(val_txt_path.resolve()),
    }
    data_yaml_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    print("\n[SUCCESS] Balanced dataset creation complete!")
    print(f"  - Labels directory      : {labels_dir.resolve()}")
    print(f"  - Train split file      : {train_txt_path.resolve()} ({len(train_paths)} images)")
    print(f"  - Validation split file : {val_txt_path.resolve()} ({len(val_paths)} images)")
    print(f"  - YAML config file      : {data_yaml_path.resolve()}")

if __name__ == "__main__":
    main()