# ocr.py
# Usage:
#   python ocr.py gemma3 data/test_images
#   python ocr.py gemma3 data/test_images --recursive
#   python ocr.py gemma3 data/test_images --prompt "Extract all readable text. Return plain text only."

import json
import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Any

from ollama import chat

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
#DEFAULT_PROMPT = "Extract all readable text from this image. Return plain text only."
DEFAULT_PROMPT = """You are a document layout analysis expert. Your task is ONLY to detect the structural layout of this form — do NOT extract field values or data.
Identify all structural elements:
- Section containers (headers, fieldsets, tables, signature blocks, free text areas)
- Individual fields within each section (their label, type, and bounding box)
- Legends, code tables, or footnotes that define coded values used in form fields

For bounding boxes, use normalized coordinates (0.0–1.0) relative to page dimensions:
{ "x1": left, "y1": top, "x2": right, "y2": bottom }

Return ONLY this JSON structure — no values, no extracted data:
{
  "form_title": "...",
  "layout_type": "...",
  "legends": [
    {
      "legend_id": "...",         ← snake_case identifier, e.g. "species_codes"
      "label": "...",             ← display name as it appears on the form
      "bounding_box": { "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0 },
      "entries": [
        { "code": "...", "label": "..." }
      ]
    }
  ],
  "notes": [
    {
      "note_id": "...",        ← snake_case identifier, e.g. "general_notes", "reporting_instructions"
      "label": "...",          ← heading as it appears on the form, e.g. "NOTES", "INSTRUCTIONS"
      "bounding_box": { "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0 },
      "content": "..."         ← full verbatim text of the note block
    }
  ],
  "sections": [
    {
      "section_name": "...",
      "section_type": "...",
      "bounding_box": { "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0 },
      "fields": [
        {
          "label": "...",
          "field_type": "...",
          "bounding_box": { "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0 },
          "is_filled": true/false,
          "legend_ref": "..."     ← optional: reference to a legend_id if this field uses coded values
        }
      ]
    }
  ]
}

LEGEND DETECTION RULES:
- Legends may appear anywhere on the form: header, footer, sidebar, or within a section
- Capture any key/code table, abbreviation list, or footnote that maps codes or symbols to labels
- Assign a descriptive snake_case legend_id (e.g. "species_codes", "gear_type_codes", "area_codes")
- Extract all code→label pairs visible in the legend
- If a field's expected values are governed by a legend, set its "legend_ref" to the matching legend_id
- If no legends are present on the form, return an empty array: "legends": []

NOTE DETECTION RULES:
- Capture any block of instructional, explanatory, or regulatory text that is not a fillable field
- Notes typically appear under headings like "NOTES", "INSTRUCTIONS", "IMPORTANT", or as numbered/lettered items
- Assign a descriptive snake_case note_id (e.g. "general_notes", "reporting_instructions", "legal_disclaimer")
- Transcribe the full verbatim text into "content", preserving numbered or lettered list structure if present
- Notes are distinct from legends: legends map codes to labels; notes are free-form explanatory text
- If no notes are present on the form, return an empty array: "notes": []"""

# Finds occurrences like "_page12" or "-page12" anywhere in the stem
PAGE_PAT = re.compile(r"(?i)([_-])page(?P<page>\d+)\b")

# Same log file used by llm_extract_core.py
RAW_OLLAMA_LOG = Path("ollama_raw_calls.jsonl")


def _resp_to_jsonable(resp: Any) -> Any:
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:
            pass
    if isinstance(resp, dict):
        return resp
    return str(resp)


def _extract_content(resp: Any) -> str:
    if hasattr(resp, "message") and hasattr(resp.message, "content"):
        return resp.message.content or ""
    if isinstance(resp, dict):
        return (resp.get("message") or {}).get("content", "") or ""
    return str(resp)


def _log_ollama_call(tag: str, model: str, payload: dict, resp: Any) -> None:
    raw_response = _resp_to_jsonable(resp)
    response_content = _extract_content(resp)

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "tag": tag,
        "model": model,
        "request": payload,
        "response": raw_response,
        "response_content": response_content,
    }

    print(f"\n===== {tag} =====")
    try:
        print(json.dumps(record, indent=2, ensure_ascii=False)[:30000])
    except Exception:
        print(record)
    sys.stdout.flush()

    with RAW_OLLAMA_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ocr_image(model: str, image_path: Path, prompt: str) -> str:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        "options": {"temperature": 0.0},
        "stream": False,
    }

    resp = chat(
        model=model,
        messages=payload["messages"],
        options=payload["options"],
        stream=payload["stream"],
    )

    _log_ollama_call(
        tag="ocr_image",
        model=model,
        payload={
            "image_path": str(image_path),
            "prompt": prompt,
            **payload,
        },
        resp=resp,
    )

    return _extract_content(resp)


def doc_key_and_page(stem: str) -> tuple[str, int]:
    """
    Robustly returns (doc_key, page_num).

    Works for:
      Doc_page1
      Doc_page01
      Doc_page1_rotated
      Doc-page2-anything
      Doc_page3 (1)
    by taking the LAST occurrence of *_page<digits>* in the filename stem.
    """
    last = None
    for m in PAGE_PAT.finditer(stem):
        last = m

    if not last:
        return stem, 0

    page_num = int(last.group("page"))
    doc = stem[: last.start()]  # everything before "_pageN"
    doc = re.sub(r"[_-]+$", "", doc).strip()  # trim trailing separators
    if not doc:
        doc = stem  # fallback safety
    return doc, page_num


def main():
    if len(sys.argv) < 3:
        print("Usage: python ocr.py <model> <images_folder> [--recursive] [--prompt <text>]")
        sys.exit(1)

    model = sys.argv[1]
    in_dir = Path(sys.argv[2]).resolve()
    recursive = "--recursive" in sys.argv

    prompt = DEFAULT_PROMPT
    if "--prompt" in sys.argv:
        i = sys.argv.index("--prompt")
        if i + 1 < len(sys.argv):
            prompt = sys.argv[i + 1]

    if not in_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {in_dir}")

    out_dir = Path(str(in_dir) + "_ocrtext")
    out_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        images = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        images = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    if not images:
        print(f"No images found in {in_dir} (extensions: {sorted(IMAGE_EXTS)})")
        sys.exit(0)

    groups = defaultdict(list)
    for img in images:
        doc_key, page_num = doc_key_and_page(img.stem)
        groups[doc_key].append((page_num, img))

    doc_names = sorted(groups.keys())

    ok_docs = 0
    fail_pages = 0
    total_pages = 0

    for doc in doc_names:
        pages = sorted(groups[doc], key=lambda x: (x[0], x[1].name))
        merged_parts = []

        for page_num, img in pages:
            total_pages += 1
            try:
                text = ocr_image(model=model, image_path=img, prompt=prompt).strip()

                if page_num > 0:
                    merged_parts.append(f"\n\n===== PAGE {page_num} ({img.name}) =====\n{text}\n")
                else:
                    merged_parts.append(f"\n\n===== IMAGE ({img.name}) =====\n{text}\n")

                print(f"[OK] {doc} :: {img.name}")
            except Exception as e:
                fail_pages += 1
                merged_parts.append(
                    f"\n\n===== PAGE {page_num if page_num > 0 else '?'} ({img.name}) OCR FAILED =====\n{e}\n"
                )
                print(f"[FAIL] {doc} :: {img} -> {e}")

        out_path = out_dir / f"{doc}.txt"
        out_path.write_text("".join(merged_parts).lstrip(), encoding="utf-8")
        ok_docs += 1
        print(f"[MERGED] {doc} -> {out_path.name} ({len(pages)} page(s))")

    print("\nDone.")
    print(f"Documents merged: {ok_docs}")
    print(f"Pages processed: {total_pages}")
    print(f"Page failures: {fail_pages}")
    print(f"Output folder: {out_dir}")
    print(f"Raw Ollama log: {RAW_OLLAMA_LOG.resolve()}")


if __name__ == "__main__":
    main()