#!/usr/bin/env bash
set -euo pipefail

# 1) 가상환경은 사용자 선택. 여기서는 시스템 파이썬 기준
# 2) .env 로드 (Gemini API 키 등)
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

echo "== Wheel City AI 2 :: Pipeline start =="

# 0) 디렉토리 준비
mkdir -p input_images bbox_images results data/raw

# 1) (사용자) data/raw 에 zenodo 데이터 압축 해제 후 위치시켜 주세요.
#    예: data/raw/wm_barriers_data/ ...  (annotations.xml 포함)
#    아래 스텝은 해제되어 있다고 가정합니다.

# 2) XML→YOLO 변환 + split 생성
python3 yolov8/prepare_dataset.py

# 3) 리밸런싱 (기본 1:1; 1:2 원하면 --ratio 2.0)
python3 yolov8/balance_dataset.py --ratio 1.0

# 4) YOLO 학습 (balanced split 사용)
python3 yolov8/train.py --balance

# 5) YOLO 추론 (input_images → bbox_images)
python3 yolov8/run.py

# 6) Gemini 추론 (bbox_images → results/result.json)
python3 gemini/run.py

echo "== Done =="
echo " - BBox images: $(realpath bbox_images)"
echo " - Final JSON : $(realpath results/result.json)"
