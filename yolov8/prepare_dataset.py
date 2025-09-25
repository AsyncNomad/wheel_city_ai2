#!/usr/bin/env python3
# yolov8/prepare_dataset.py
import argparse
from pathlib import Path
from lxml import etree
from tqdm import tqdm
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# CVAT 속성 카테고리 → 하한값 매핑
STEP_HEIGHT_MAP = {
    "less than 3cm": 0,
    "3cm to 7cm": 3,
    "more than 7cm": 7,
}
RAMP_WIDTH_MAP = {
    "less than 50cm": 0,
    "50cm to 100cm": 50,
    "more than 100cm": 100,
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="yolov8/images")
    ap.add_argument("--xml_glob",   default="yolov8/**/*.xml")
    ap.add_argument("--labels_dir", default="yolov8/labels")
    ap.add_argument("--val_ratio",  type=float, default=0.1)
    # 속성 필터(옵션)
    ap.add_argument("--min_step_height", type=int, default=None, help="0/3/7 권장. None이면 필터 비활성")
    ap.add_argument("--min_ramp_width",  type=int, default=None, help="0/50/100 권장. None이면 필터 비활성")
    return ap.parse_args()

def get_attr_map(box_elem):
    """CVAT <box><attribute name="...">VALUE</attribute>...</box> -> dict"""
    out = {}
    for a in box_elem.findall(".//attribute"):
        name = (a.get("name") or "").strip().lower()
        val  = (a.text or "").strip().lower()
        if name:
            out[name] = val
    return out

def pass_filters(label, attrs, min_step_height, min_ramp_width):
    """데이터셋 설명의 속성 조건을(선택적으로) 반영"""
    if min_step_height is not None and label == "step":
        cat = attrs.get("height") or attrs.get("step height") or ""
        h = STEP_HEIGHT_MAP.get(cat, None)
        if h is None or h < min_step_height:
            return False
    if min_ramp_width is not None and label == "ramp":
        cat = attrs.get("width") or attrs.get("ramp width") or ""
        w = RAMP_WIDTH_MAP.get(cat, None)
        if w is None or w < min_ramp_width:
            return False
    return True

def xml_to_yolo(xml_path: Path, images_root: Path, labels_dir: Path,
                min_step_height, min_ramp_width) -> int:
    tree = etree.parse(str(xml_path))
    kept = 0
    for im in tree.findall(".//image"):
        name = im.get("name") or ""
        if not name:
            continue
        # 이미지 경로 결정(이름만 일치해도 하위 탐색)
        img = images_root / Path(name).name
        if not img.exists():
            cands = list(images_root.rglob(Path(name).name))
            if cands:
                img = cands[0]
        if not img.exists() or img.suffix.lower() not in IMG_EXTS:
            continue

        try:
            W = float(im.get("width", 0)); H = float(im.get("height", 0))
        except Exception:
            W = H = 0
        if W <= 0 or H <= 0:
            continue

        lines = []
        for box in im.findall(".//box"):
            label = (box.get("label") or "").strip().lower()
            if label not in ("step", "stair", "ramp"):
                # grab bar 등은 제외
                continue
            attrs = get_attr_map(box)
            if not pass_filters(label, attrs, min_step_height, min_ramp_width):
                continue
            try:
                xtl=float(box.get("xtl")); ytl=float(box.get("ytl"))
                xbr=float(box.get("xbr")); ybr=float(box.get("ybr"))
            except Exception:
                continue
            if not (xbr > xtl and ybr > ytl):
                continue
            cx = ((xtl + xbr) / 2.0) / W
            cy = ((ytl + ybr) / 2.0) / H
            bw = (xbr - xtl) / W
            bh = (ybr - ytl) / H
            if bw <= 0 or bh <= 0:
                continue
            cls = 0 if label == "ramp" else 1  # 0:ramp, 1:barrier(step+stair)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if lines:
            labels_dir.mkdir(parents=True, exist_ok=True)
            (labels_dir / f"{img.stem}.txt").write_text("".join(lines), encoding="utf-8")
            kept += 1
    return kept

def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    assert images_dir.exists(), "yolov8/images 가 존재해야 합니다."

    # 1) XML→YOLO 라벨 생성(없어도 전체 스플릿엔 포함됨)
    xmls = sorted(Path(".").glob(args.xml_glob))
    total_labeled = 0
    for x in tqdm(xmls or [], desc="Parsing XML"):
        total_labeled += xml_to_yolo(
            x, images_dir, labels_dir, args.min_step_height, args.min_ramp_width
        )

    # 2) 전체 이미지 기준 스플릿 생성
    all_imgs = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    assert all_imgs, "yolov8/images 아래 이미지가 없습니다."

    n = len(all_imgs)
    n_val = int(n * args.val_ratio)
    val = all_imgs[:n_val]
    tr  = all_imgs[n_val:]

    yolov8_dir = Path("yolov8")
    (yolov8_dir / "train.txt").write_text(
        "\n".join(str(p.resolve()) for p in tr) + "\n", encoding="utf-8"
    )
    (yolov8_dir / "val.txt").write_text(
        "\n".join(str(p.resolve()) for p in val) + "\n", encoding="utf-8"
    )

    data = {
        "names": {0: "ramp", 1: "barrier"},
        "train": str((yolov8_dir / "train.txt").resolve()),
        "val":   str((yolov8_dir / "val.txt").resolve()),
    }
    (yolov8_dir / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    print(f"[OK] labels dir : {labels_dir.resolve()}")
    print(f"[OK] splits     : { (yolov8_dir/'train.txt').resolve() } , { (yolov8_dir/'val.txt').resolve() }")
    print(f"[OK] data.yaml  : { (yolov8_dir/'data.yaml').resolve() }")
    print(f"[INFO] images total: {n}, labeled(from XML): {total_labeled}")

if __name__ == "__main__":
    main()
