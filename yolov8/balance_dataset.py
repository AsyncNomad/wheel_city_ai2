#!/usr/bin/env python3
# yolov8/balance_dataset.py
import argparse, random, shutil
from pathlib import Path
from tqdm import tqdm
import yaml

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split", default="yolov8/train.txt")
    ap.add_argument("--val_split",   default="yolov8/val.txt")
    ap.add_argument("--labels_dir",  default="yolov8/labels")
    ap.add_argument("--barrier_per_ramp",  type=float, default=1.0, help="barrier : ramp 비율")
    ap.add_argument("--negative_per_ramp", type=float, default=1.0, help="negative : ramp 비율")
    # 최종 학습용 서브셋 위치(요청사항)
    ap.add_argument("--out_root",    default="yolov8/train")
    return ap.parse_args()

def read_list(p: Path):
    return [Path(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def label_path(img: Path, labels_dir: Path):
    return labels_dir / f"{img.stem}.txt"

def category_of(img: Path, labels_dir: Path) -> str:
    """
    ramp: '0 ' 포함
    barrier: '1 '은 있으나 '0 '은 없음
    negative: 라벨 미존재 or 빈 파일 or 둘 다 없음
    """
    lp = label_path(img, labels_dir)
    if not lp.exists():
        return "negative"
    has0 = False; has1 = False; any_line = False
    for line in lp.read_text(encoding="utf-8").splitlines():
        sp = line.strip().split()
        if not sp: continue
        any_line = True
        c = sp[0]
        if c == "0": has0 = True
        elif c == "1": has1 = True
    if has0:
        return "ramp"
    if has1 and not has0:
        return "barrier"
    if (not any_line) or (not has0 and not has1):
        return "negative"
    return "negative"

def choose_subset(imgs, labels_dir: Path, bpr: float, npr: float):
    ramps, barriers, negatives = [], [], []
    for im in imgs:
        cat = category_of(im, labels_dir)
        (ramps if cat=="ramp" else barriers if cat=="barrier" else negatives).append(im)
    random.shuffle(ramps); random.shuffle(barriers); random.shuffle(negatives)

    n_r = len(ramps)
    n_b = min(len(barriers), int(n_r * bpr))
    n_n = min(len(negatives), int(n_r * npr))
    chosen = ramps + barriers[:n_b] + negatives[:n_n]
    random.shuffle(chosen)

    stats = {
        "ramps": len(ramps), "barriers": len(barriers), "negatives": len(negatives),
        "kept_ramps": len(ramps), "kept_barriers": n_b, "kept_negatives": n_n,
        "total": len(chosen)
    }
    return chosen, stats

def copy_subset(chosen, labels_dir: Path, out_img_dir: Path, out_lbl_dir: Path):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for im in tqdm(chosen, desc=f"Copying -> {out_img_dir}"):
        dst_im = out_img_dir / im.name
        if not dst_im.exists():
            shutil.copy2(im, dst_im)
        src_lbl = label_path(im, labels_dir)
        dst_lbl = out_lbl_dir / f"{im.stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            # negative: 빈 라벨 파일 생성
            if not dst_lbl.exists():
                dst_lbl.write_text("", encoding="utf-8")
        written.append(dst_im.resolve())
    return written

def main():
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    train_imgs = read_list(Path(args.train_split))
    val_imgs   = read_list(Path(args.val_split))

    # 최종 서브셋 경로(요청사항): yolov8/train/images, yolov8/train/labels
    out_img_dir = Path(args.out_root) / "images"
    out_lbl_dir = Path(args.out_root) / "labels"

    tr_chosen, tr_stats = choose_subset(train_imgs, labels_dir, args.barrier_per_ramp, args.negative_per_ramp)
    vl_chosen, vl_stats = choose_subset(val_imgs,   labels_dir, args.barrier_per_ramp, args.negative_per_ramp)

    tr_written = copy_subset(tr_chosen, labels_dir, out_img_dir, out_lbl_dir)
    vl_written = copy_subset(vl_chosen, labels_dir, out_img_dir, out_lbl_dir)  # 검증도 같은 폴더에 모음

    yolov8_dir = Path("yolov8")
    (yolov8_dir / "train_balanced.txt").write_text(
        "\n".join(str(p) for p in tr_written) + "\n", encoding="utf-8"
    )
    (yolov8_dir / "val_balanced.txt").write_text(
        "\n".join(str(p) for p in vl_written) + "\n", encoding="utf-8"
    )

    data_bal = {
        "names": {0: "ramp", 1: "barrier"},
        "train": str((yolov8_dir / "train_balanced.txt").resolve()),
        "val":   str((yolov8_dir / "val_balanced.txt").resolve()),
    }
    (yolov8_dir / "data_balanced.yaml").write_text(
        yaml.safe_dump(data_bal, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    print(f"[STATS][train] ramps:{tr_stats['ramps']} barriers:{tr_stats['barriers']} negatives:{tr_stats['negatives']} -> kept r:{tr_stats['kept_ramps']} b:{tr_stats['kept_barriers']} n:{tr_stats['kept_negatives']} total:{tr_stats['total']}")
    print(f"[STATS][val]   ramps:{vl_stats['ramps']} barriers:{vl_stats['barriers']} negatives:{vl_stats['negatives']} -> kept r:{vl_stats['kept_ramps']} b:{vl_stats['kept_barriers']} n:{vl_stats['kept_negatives']} total:{vl_stats['total']}")
    print(f"[OK] subset images: {out_img_dir.resolve()}")
    print(f"[OK] subset labels: {out_lbl_dir.resolve()}")
    print(f"[OK] splits/data : { (yolov8_dir/'train_balanced.txt').resolve() } , { (yolov8_dir/'val_balanced.txt').resolve() } , { (yolov8_dir/'data_balanced.yaml').resolve() }")

if __name__ == "__main__":
    main()
