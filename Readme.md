# 개선된 휠체어 접근성 분석 AI - Wheel City AI 2

`Wheel City AI 2`는 건물 입구 이미지 한 장으로 휠체어 접근성을 자동으로 분석하고 판단하는 딥러닝 기반 프로젝트입니다. **YOLOv8**의 객체 탐지 기술과 **Gemini**의 상황 인지 능력을 결합하여, 사진 속 장소에 대해 이동 약자의 통행 가능 여부를 판단합니다.

## 프로젝트 메커니즘

이 프로젝트는 두 가지 AI 모델이 유기적으로 협력하는 파이프라인 구조로 동작합니다.

1. **1단계: 객체 탐지 (YOLOv8m)**
    - 사용자가 `test_images` 폴더에 이미지를 입력하면, 사전 학습된 **YOLOv8m 모델**이 먼저 작동합니다.
    - 모델은 이미지 내에서 휠체어 접근성의 핵심 요소인 턱/계단(curb)과 경사로(ramp)를 탐지합니다.
    - 탐지된 객체에는 바운딩 박스(Bounding Box)가 표시되며, 이 시각화된 이미지는 LLaVA-1.6에 input으로 전달됩니다.
2. **2단계: 종합 판단 (Gemini 2.5 Flash / Gemini 2.5 Pro)**
    - 1단계에서 생성된 바운딩 박스 이미지를 **Gemini**가 입력받습니다.
    - Gemini는 단순 객체 유무를 넘어, "턱이 있지만, 문으로 이어지는 유효한 경사로가 있는가?"와 같이 **이미지의 전체적인 맥락과 상황을 종합적으로 이해**하고 추론합니다.
    - 최종적으로, Gemini는 접근성 규칙에 기반하여 `accessible` (접근 가능 여부), `reason` (판단 이유)이 포함된 구조화된 **JSON 형식의 최종 결과**를 생성합니다.

---

## 사용 모델 (Models Used)

| 역할 | 모델 이름 | 상세 정보 |
| --- | --- | --- |
| **객체 탐지** | YOLOv8 | Ultralytics의 작고 빠른 객체 탐지 모델로 로컬에서 동작 |
| **종합 판단** | Gemini 2.5 Flash / Pro | Google AI Studio를 통해 API 호출로 동작 |

---

## 디렉토리 구조

```bash
wheel_city_ai/
├── yolov8/                   # YOLOv8 모델을 위한 프로젝트 폴더
│   ├── run.py                # step 2. YOLOv8 분석을 실행하는 스크립트
│   └── ...                   # (학습 데이터, 모델 가중치 등)
│
├── gemini/                   # gemini 모델을 위한 프로젝트 폴더
│   ├── run.py                # step 3. gemini 분석을 실행하는 스크립트
│   └── ...                   
├── runner/                   # 전체 과정 실행 프로그램을 위한 프로젝트 폴더
│   ├── main.rs               # 전체 과정을 자동화한 최종 실행 프로그램 코드
│   └── ...                   # (모델 가중치 등)
│
├── input_images/           # step 1. 입력할 사진을 넣는 디렉토리
│── bbox_images/            # YOLO의 분석 결과 사진이 임시 저장되는 디렉토리
└── results/                # 결과 json 파일이 저장되는 디렉토리
```

---

## 사용 방법

1. **이미지 입력:**
    - `test_images/` 폴더에 분석하고 싶은 건물 입구 이미지를 넣습니다.
2. **YOLOv8 실행 (객체 탐지):**
    - `yolov8` 폴더로 이동하여 `run.py` 스크립트를 실행합니다.
    - 실행이 완료되면 `bbox_images/` 폴더에 바운딩 박스가 표시된 이미지들이 생성됩니다.
3. **Gemini 실행 (최종 판단):**
    - `gemini/` 폴더로 이동하여 `run.py` 스크립트를 실행합니다.
4. **결과 확인:**
    - 최종 분석 결과는 `results/` 폴더 안의 `result.json` 파일에서 확인할 수 있습니다.

---

## 실행 예시

1. input 이미지 준비

 <img src="https://github.com/user-attachments/assets/58d210dc-d75a-4fa5-b0d0-7a3e4660ed41" width="400" height="600"/>

2. YOLOv8s가 턱과 경사로를 감지하여 핀

 <img src="https://github.com/user-attachments/assets/f514953d-f931-42ca-84d9-ed8f292fa24a" width="400" height="600"/>

3. Gemini가 상황을 판단하여 최종 의사결정, 스크립트를 통해 JSON으로 파싱

```json
{
  "results": [
    {
      "image": "annotated_data1.jpg",
      "result": {
        "accessible": true,
        "reason": "The building entrance has a permanent ramp connecting the ground to the entrance, making it accessible for a lone wheelchair user despite the presence of curbs."
      }
    }
  ]
}
```

---

## 🛠️ 환경 설정 (Ubuntu 24.04 기준)
실행 전 .env 반드시 수정하기. 본인의 API KEY를 설정하고, YOLOv8 모델 학습 설정 커스텀할 수 있음.
```bash
# Google Gemini
GOOGLE_API_KEY="여기 자신의 API KEY 넣기"

# 사용할 YOLO 모델 (n, s, m, l, x 중 선택)
YOLO_MODEL=yolov8m.pt

# 데이터셋 yaml
YOLO_DATA=yolov8/data_balanced.yaml

# 학습 설정
EPOCHS=50
IMGSZ=640
BATCH=8

# 결과 저장 폴더 이름
RUN_NAME=ver1
```