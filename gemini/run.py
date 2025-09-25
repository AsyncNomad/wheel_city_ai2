#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# ===== Prompt =====
SYSTEM_PROMPT = (
    "You are an accessibility analysis AI. Analyze the provided image of a building entrance to determine if it is accessible for a lone wheelchair user.\n"
    "Accessibility Rules:\n"
    "1. There must be no steps or curbs between the ground and the entrance.\n"
    "2. If there are steps or curbs, a ramp must connect the ground to the entrance.\n\n"
    "Return ONLY valid JSON. Do not include any explanations, Markdown, or code fences.\n"
    'JSON schema: {"accessible": boolean | null, "reason": string}\n'
)

# gemini does not support jpg. mapping jpg -> image/jpeg
MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
SUPPORTED_EXTS = set(MIME_BY_EXT.keys())

def guess_mime(path: Path) -> str | None:
    return MIME_BY_EXT.get(path.suffix.lower())

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="bbox_images", help="분석할 이미지 폴더")
    ap.add_argument("--out_json",   default="results/result.json", help="결과 JSON 저장 경로")
    ap.add_argument("--model",      default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--timeout",    type=float, default=60.0)
    return ap.parse_args()

# -------- Robust JSON extraction --------
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
BRACE_SPAN_RE = re.compile(r"\{.*\}", re.DOTALL)

def try_extract_json(text: str) -> str | None:
    if not text:
        return None
    # 1) ```json ... ``` 우선
    m = JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # 2) 가장 바깥 {...} 블록 추정
    #    (간단한 휴리스틱: 첫 '{'와 마지막 '}'를 찾아 자르기)
    first = text.find("{")
    last  = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last+1].strip()
        # 후보가 JSON처럼 보이면 반환
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    # 3) 폴백: 정규식으로 아무 {...} 매칭
    m2 = BRACE_SPAN_RE.search(text)
    if m2:
        return m2.group(0).strip()
    return None

def safe_json(text: str) -> dict:
    """모델 응답 텍스트에서 JSON을 최대한 추출/보정."""
    # 0) 바로 파싱 시도
    for candidate in (text, try_extract_json(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise ValueError("not a dict")
            # 보정
            acc = obj.get("accessible", None)
            if acc not in (True, False, None):
                acc = None
            reason = obj.get("reason", "No reason provided.")
            if not isinstance(reason, str):
                reason = str(reason)
            return {"accessible": acc, "reason": reason}
        except Exception:
            pass
    # 실패
    return {"accessible": None, "reason": "Parse error: model did not return valid JSON."}

def main():
    load_dotenv()
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set (.env).")

    genai.configure(api_key=api_key)

    # 시스템 프롬프트를 적용하고 JSON 모드로 강제
    # (일부 버전에서만 지원되지만, 미지원이면 무시되며 문제 없음)
    model = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "response_mime_type": "application/json",
            # "temperature": 0.2,  # 원하면 추가
        },
        # 필요시 안전 설정 완화 가능 (과도 차단 방지)
        # safety_settings={"HARASSMENT": "BLOCK_NONE", ...}
    )

    images_dir = Path(args.images_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    if not files:
        raise RuntimeError(f"No images found under {images_dir} (supported: {sorted(SUPPORTED_EXTS)})")

    results = []
    for img_path in files:
        try:
            mime = guess_mime(img_path)
            if not mime:
                results.append({
                    "image": img_path.name,
                    "result": {"accessible": None, "reason": f"Unsupported extension: {img_path.suffix}"}
                })
                continue

            img_bytes = img_path.read_bytes()

            # JSON 강제 응답을 기대하되, 혹시 그래도 텍스트가 섞이면 safe_json이 처리
            resp = model.generate_content(
                [{"inline_data": {"mime_type": mime, "data": img_bytes}}],
                request_options={"timeout": args.timeout},
            )

            # .text가 있으면 사용, 없으면 candidates에서 복구
            text = getattr(resp, "text", None)
            if not text:
                try:
                    cand = resp.candidates[0]
                    parts = getattr(cand.content, "parts", [])
                    text = "".join(getattr(p, "text", "") for p in parts)
                except Exception:
                    text = ""

            result_obj = safe_json((text or "").strip())
            results.append({"image": img_path.name, "result": result_obj})

        except Exception as e:
            results.append({
                "image": img_path.name,
                "result": {"accessible": None, "reason": f"Request error: {e}"}
            })

    payload = {"results": results}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved: {out_json.resolve()}")

if __name__ == "__main__":
    main()
