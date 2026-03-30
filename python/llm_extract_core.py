from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime
from ollama import chat


MODEL_CANNOT_SCORE = "model is not capable of giving confidence score"
DEBUG_OLLAMA = True

'''
SYSTEM_PROMPT = """You are an information extraction assistant.

You will receive:
- A context note
- The full document text (or a chunk)
- A JSON list of questions with ids

Return ONLY valid JSON, and ONLY this structure:
{
  "answers": {
    "<id>": <string or null>,
    ...
  }
}

Rules:
- Use only evidence from the document text.
- If not present / not inferable, use null.
- Do not add extra keys.
- No markdown, no explanation.
"""
'''
SYSTEM_PROMPT = """You are an OCR data extraction expert.
You will be given:
1. A LAYOUT MAP (JSON) from a prior layout detection step, containing
   sections, fields, field types, bounding boxes, is_filled flags,
   optional legend_ref references, and a top-level legends array
2. The original form image

Your task: extract the VALUE of every field where is_filled = true,
using the bounding_box coordinates to locate each field precisely
in the image before reading it.

LAYOUT MAP: given as JSON, already extracted by a prior layout detection step. It contains a "sections" array, where each section has a "fields" array.

=== EXTRACTION RULES ===

GENERAL:
- Use bounding_box coordinates to locate each field in the image
  before extracting its value
- Only extract fields where is_filled = true
- Set value = null for all fields where is_filled = false
- Preserve section hierarchy from the layout map in your output
- Do not infer or guess values — only transcribe what is visible
- Do NOT re-extract or expand legends — they are already captured
  in the layout map and must not appear in this output

FIELD TYPE RULES:
- text_input / number_input: transcribe exactly as printed
- segmented_text_input (e.g. VRN, License #): return as single
  concatenated string, e.g. "123456"
- date_input / segmented_date_input: return as YYYY-MM-DD if
  determinable, otherwise transcribe as-is
- time_input_24hr: return as HH:MM
- checkbox: return true if checked/marked, false if empty
- radio_group / circled_option: return the selected option label
  as a string
- signature: return { "present": true, "handwritten": true }
- text_label / section_label / table_column_header: skip —
  these are structural, not data fields
- table_row / table_rows: return as array of objects using
  column headers as keys (see TABLE RULES)

HANDWRITTEN TEXT:
- Flag any handwritten value with "handwritten": true
- Transcribe exactly as written, including abbreviations

CONFIDENCE LEVELS:
- "high": clearly printed/typed, unambiguous
- "medium": partially legible, or inferred from context
- "low": unclear, obscured, ambiguous, or conflicting

TABLE RULES:
- Return each table as an array of row objects
- Use column header labels as keys
- Example:
  {
    "table_name": "TRAP STRING HAULED LOG",
    "rows": [
      {
        "Date": "2021/04/02",
        "Latitude": "47°03.10'N",
        "Longitude": "60°22.90'W",
        "Depth (fath.)": "75",
        "# of traps": "20",
        "Soak time (days)": "2",
        "Catch Kept (pounds)": "1000",
        "Comments": "bycatch of 2 scuplin"
      }
    ]
  }

=== OUTPUT FORMAT ===
{
  "form_title": "...",
  "layout_template_id": "...",
  "extracted_at": "ISO-8601 timestamp",
  "sections": [
    {
      "section_name": "...",
      "fields": [
        {
          "label": "...",
          "value": "...",
          "field_type": "...",
          "handwritten": false,
          "confidence": "high | medium | low"
        }
      ],
      "tables": [
        {
          "table_name": "...",
          "handwritten": "...",
          "rows": [ { "column_header": "value" } ]
        }
      ]
    }
  ],
  "flags": [
    {
      "type": "missing_required | low_confidence | illegible | unexpected_format | cross_field_inconsistency",
      "message": "Human-readable description of the issue",
      "field_label": "...",
      "section_name": "...",
      "bounding_box": { "x": 0, "y": 0, "width": 0, "height": 0 }
    }
  ]
}

FLAGS RULES:
Raise a flag for any of the following: missing required fields,
low-confidence extractions, illegible values, unexpected formats,
or cross-field inconsistencies (e.g. Date Sailed is after Date Landed)
For each flag, include:

"type": one of the categories above
"message": a clear, human-readable description of the issue
"field_label": the label of the affected field, carried forward
from the layout map
"section_name": the section containing the affected field,
carried forward from the layout map
"bounding_box": the bounding_box of the affected field copied
exactly from the layout map; omit this key only if no
bounding_box exists in the layout map for that field

For cross-field inconsistencies involving two fields, include the
bounding boxes of both under "bounding_boxes" (array) instead of
the singular "bounding_box" key
"""

EVIDENCE_SYSTEM_PROMPT = f"""You are an information extraction assistant.

You will receive:
- A context note
- The full document text (or a chunk)
- A JSON list of questions with ids

Return ONLY valid JSON, and ONLY this structure:
{{
  "answers": {{
    "<id>": {{
      "value": <string or null>,
      "confidence": <number between 0 and 1 or null>,
      "confidence_note": <string or null>,
      "excerpt_location": {{
        "page": <integer or null>,
        "sentence": <string or null>
      }}
    }}
  }}
}}

Rules:
- Use only evidence from the document text.
- If not present / not inferable, use null for value.
- page must come from explicit page markers in the document text, such as:
  ===== PAGE 7 =====
- sentence must be copied from the document text if provided.
- Do not invent or estimate confidence scores in Python.
- If the model cannot provide a true confidence score, set confidence to null and confidence_note to "{MODEL_CANNOT_SCORE}".
- Do not add extra keys.
- No markdown, no explanation.
"""

_PAGE_MARKER_RE = re.compile(
    r"^=+\s*PAGE\s+(\d+)\b[^\n]*=+\s*$",
    flags=re.MULTILINE | re.IGNORECASE,
)

RAW_OLLAMA_LOG = "ollama_raw_calls.jsonl"


def _resp_to_jsonable(resp: Any) -> Any:
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:
            pass
    if isinstance(resp, dict):
        return resp
    return str(resp)


def _log_ollama_call(tag: str, model: str, payload: dict, resp: Any) -> None:
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "tag": tag,
        "model": model,
        "request": payload,
        "response": _resp_to_jsonable(resp),
        "response_content": _extract_chat_content(resp),
    }

    print(f"\n===== {tag} =====")
    try:
        print(json.dumps(record, indent=2, ensure_ascii=False)[:30000])
    except Exception:
        print(record)
    sys.stdout.flush()

    with open(RAW_OLLAMA_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _debug_print(label: str, data: Any, max_chars: int = 20000) -> None:
    if not DEBUG_OLLAMA:
        return
    print(f"\n===== {label} =====")
    try:
        if isinstance(data, (dict, list)):
            txt = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            txt = str(data)
        print(txt[:max_chars] + ("\n...[truncated]" if len(txt) > max_chars else ""))
    except Exception as e:
        print(f"[debug print failed] {e}")
    sys.stdout.flush()


def _extract_chat_content(resp: Any) -> str:
    content = getattr(getattr(resp, "message", None), "content", None)
    if content is not None:
        return content or ""
    if isinstance(resp, dict):
        return (resp.get("message") or {}).get("content", "") or ""
    return str(resp)


def _extract_logprobs(resp: Any) -> Any:
    if hasattr(resp, "logprobs"):
        try:
            return resp.logprobs
        except Exception:
            pass
    if isinstance(resp, dict):
        return resp.get("logprobs")
    if hasattr(resp, "model_dump"):
        try:
            dumped = resp.model_dump()
            if isinstance(dumped, dict):
                return dumped.get("logprobs")
        except Exception:
            pass
    return None


def extract_json_object(text: str) -> Optional[str]:
    """Pull a JSON object out of markdown/codefences/prose."""
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()

    return None


def safe_load_json_from_model(text: str) -> Dict[str, Any]:
    snippet = extract_json_object(text)
    if not snippet:
        return {}
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _repair_to_json(model: str, bad_text: str) -> Dict[str, Any]:
    repair_system = "You output ONLY valid JSON. No markdown. No extra text."

    payload = {
        "messages": [
            {"role": "system", "content": repair_system},
            {"role": "user", "content": f"Fix this into valid JSON only:\n\n{bad_text}"},
        ],
        "options": {"temperature": 0.0},
        "format": "json",
        "stream": False,
    }

    resp = chat(
        model=model,
        messages=payload["messages"],
        options=payload["options"],
        format=payload["format"],
        stream=payload["stream"],
    )

    _log_ollama_call(
        tag="_repair_to_json",
        model=model,
        payload=payload,
        resp=resp,
    )

    content = getattr(getattr(resp, "message", None), "content", None) or ""
    parsed = safe_load_json_from_model(content)
    return parsed if isinstance(parsed, dict) else {}

def _normalize_question_for_model(q: str) -> str:
    q2 = (q or "").strip()
    q2 = re.sub(r"\s+", " ", q2)
    q2 = q2.rstrip(":").strip()
    return q2


def _as_int(v: Any) -> int | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(float(str(v).strip()))
    except Exception:
        return None


def _as_float(v: Any) -> float | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        x = float(str(v).strip())
        return max(0.0, min(1.0, x))
    except Exception:
        return None


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _page_segments(text: str) -> list[dict[str, Any]]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_PAGE_MARKER_RE.finditer(text))
    if not matches:
        return [{"page": None, "text": text}]

    out: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append({"page": int(m.group(1)), "text": text[start:end].strip()})
    return out


def _coerce_rich_answer(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        value = raw.get("value", raw.get("answer"))
        confidence = _as_float(raw.get("confidence"))
        confidence_note = raw.get("confidence_note")
        loc = raw.get("excerpt_location") if isinstance(raw.get("excerpt_location"), dict) else {}
        page = _as_int(loc.get("page") if loc else raw.get("page"))
        sentence = loc.get("sentence") if loc else raw.get("sentence", raw.get("excerpt"))
    else:
        value = raw
        confidence = None
        confidence_note = None
        page = None
        sentence = None

    value = None if value is None or not str(value).strip() else str(value).strip()
    confidence_note = _norm_ws(str(confidence_note)) if confidence_note else None
    sentence = _norm_ws(str(sentence)) if sentence else None

    if value is not None and confidence is None and not confidence_note:
        confidence_note = MODEL_CANNOT_SCORE

    return {
        "value": value,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "excerpt_location": {
            "page": page,
            "sentence": sentence,
        },
    }


def _rich_answer_score(item: dict[str, Any]) -> float:
    if not isinstance(item, dict):
        return 0.0

    value = item.get("value")
    if value is None or not str(value).strip():
        return 0.0

    conf = _as_float(item.get("confidence")) or 0.0
    loc = item.get("excerpt_location") if isinstance(item.get("excerpt_location"), dict) else {}
    page_bonus = 0.15 if loc.get("page") is not None else 0.0
    sent_bonus = 0.20 if loc.get("sentence") else 0.0
    len_bonus = min(len(str(value).strip()), 60) / 1000.0

    return 1.0 + (conf * 0.25) + page_bonus + sent_bonus + len_bonus


def answer_questions_json(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    temperature: float = 0.1,
) -> Dict[str, Optional[str]]:
    orig_questions = [q.strip() for q in questions if q and q.strip()]
    if not orig_questions:
        return {}

    q_items = []
    id_to_key: Dict[str, str] = {}
    for i, q in enumerate(orig_questions, start=1):
        qid = f"q{i}"
        id_to_key[qid] = q
        q_items.append({"id": qid, "question": _normalize_question_for_model(q)})

    user_prompt = (
        f"Context note:\n{context_note}\n\n"
        f"Questions:\n{json.dumps(q_items, ensure_ascii=False, indent=2)}\n\n"
        "Document text:\n<doc>\n"
        f"{document_text}\n"
        "</doc>\n\n"
        "Return JSON now."
    )
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
        "format": "json",
        "stream": False,
    }

    resp = chat(
        model=model,
        messages=payload["messages"],
        options=payload["options"],
        format=payload["format"],
        stream=payload["stream"],
    )
    _log_ollama_call(
        tag="answer_questions_json",
        model=model,
        payload=payload,
        resp=resp,
    )
    _debug_print("EXTRACTION RAW RESPONSE (simple)", _resp_to_jsonable(resp))
    logprobs = _extract_logprobs(resp)
    if logprobs:
        _debug_print("EXTRACTION LOGPROBS (simple)", logprobs)
    else:
        _debug_print("EXTRACTION CONFIDENCE NOTE (simple)", MODEL_CANNOT_SCORE)

    content = _extract_chat_content(resp)
    _debug_print("EXTRACTION MESSAGE CONTENT (simple)", content)

    parsed = safe_load_json_from_model(content)
    _debug_print("EXTRACTION PARSED JSON (simple)", parsed)

    if not isinstance(parsed, dict) or not parsed:
        parsed = _repair_to_json(model, content)

    answers: dict[str, Any] = {}
    if isinstance(parsed, dict):
        if isinstance(parsed.get("answers"), dict):
            answers = parsed["answers"]
        elif any(k.startswith("q") and k[1:].isdigit() for k in parsed.keys()):
            answers = parsed

    out: Dict[str, Optional[str]] = {}
    for qid, key in id_to_key.items():
        v = answers.get(qid)
        if v is None:
            v = answers.get(key)
        if v is None:
            v = answers.get(key.rstrip(":").strip())
        out[key] = None if v is None or not str(v).strip() else str(v).strip()

    return out


def answer_questions_json_evidence(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    temperature: float = 0.1,
) -> Dict[str, Dict[str, Any]]:
    orig_questions = [q.strip() for q in questions if q and q.strip()]
    if not orig_questions:
        return {}

    q_items = []
    id_to_key: Dict[str, str] = {}
    for i, q in enumerate(orig_questions, start=1):
        qid = f"q{i}"
        id_to_key[qid] = q
        q_items.append({"id": qid, "question": _normalize_question_for_model(q)})

    user_prompt = (
        f"Context note:\n{context_note}\n\n"
        f"Questions:\n{json.dumps(q_items, ensure_ascii=False, indent=2)}\n\n"
        "Document text:\n<doc>\n"
        f"{document_text}\n"
        "</doc>\n\n"
        "Return JSON now."
    )
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
        "format": "json",
        "stream": False,
    }

    resp = chat(
        model=model,
        messages=payload["messages"],
        options=payload["options"],
        format=payload["format"],
        stream=payload["stream"],
    )
    _log_ollama_call(
        tag="answer_questions_json_evidence",
        model=model,
        payload=payload,
        resp=resp,
    )
    _debug_print("EXTRACTION RAW RESPONSE (evidence)", _resp_to_jsonable(resp))
    logprobs = _extract_logprobs(resp)
    if logprobs:
        _debug_print("EXTRACTION LOGPROBS (evidence)", logprobs)
    else:
        _debug_print("EXTRACTION CONFIDENCE NOTE (evidence)", MODEL_CANNOT_SCORE)

    content = _extract_chat_content(resp)
    _debug_print("EXTRACTION MESSAGE CONTENT (evidence)", content)

    parsed = safe_load_json_from_model(content)
    _debug_print("EXTRACTION PARSED JSON (evidence)", parsed)

    if not isinstance(parsed, dict) or not parsed:
        parsed = _repair_to_json(model, content)

    answers: dict[str, Any] = {}
    if isinstance(parsed, dict):
        if isinstance(parsed.get("answers"), dict):
            answers = parsed["answers"]
        elif any(k.startswith("q") and k[1:].isdigit() for k in parsed.keys()):
            answers = parsed

    out: Dict[str, Dict[str, Any]] = {}
    for qid, key in id_to_key.items():
        raw = answers.get(qid)
        if raw is None:
            raw = answers.get(key)
        if raw is None:
            raw = answers.get(key.rstrip(":").strip())

        item = _coerce_rich_answer(raw)
        out[key] = {
            "value": item["value"],
            "confidence": item["confidence"],
            "confidence_note": item["confidence_note"],
            "excerpt_location": {
                "page": item["excerpt_location"].get("page"),
                "sentence": item["excerpt_location"].get("sentence"),
            },
        }

    return out


def _chunk_text(s: str, max_chars: int = 12000, overlap: int = 800) -> list[str]:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    page_segs = _page_segments(s)

    if len(page_segs) == 1 and page_segs[0].get("page") is None:
        chunks: list[str] = []
        i = 0
        while i < len(s):
            j = min(len(s), i + max_chars)
            chunks.append(s[i:j])
            if j == len(s):
                break
            i = max(0, j - overlap)
        return chunks

    page_blocks: list[str] = []
    for seg in page_segs:
        page = seg.get("page")
        txt = seg.get("text", "")
        page_blocks.append(f"===== PAGE {page} =====\n{txt}".strip())

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for block in page_blocks:
        block_len = len(block) + 2
        if current and current_len + block_len > max_chars:
            chunks.append("\n\n".join(current))

            if overlap > 0:
                carry: list[str] = []
                carry_len = 0
                for old in reversed(current):
                    carry.insert(0, old)
                    carry_len += len(old) + 2
                    if carry_len >= overlap:
                        break
                current = carry
                current_len = sum(len(x) + 2 for x in current)
            else:
                current = []
                current_len = 0

        current.append(block)
        current_len += block_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def answer_questions_json_chunked(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    max_chars: int = 12000,
    overlap: int = 800,
) -> Dict[str, Optional[str]]:
    qs = [q.strip() for q in questions if q and q.strip()]
    if not qs:
        return {}

    merged: Dict[str, Optional[str]] = {q: None for q in qs}
    chunks = _chunk_text(document_text, max_chars=max_chars, overlap=overlap)

    for idx, ch in enumerate(chunks, start=1):
        _debug_print("PROCESSING CHUNK (simple)", f"{idx}/{len(chunks)}")

        partial = answer_questions_json(
            model=model,
            document_text=f"[CHUNK {idx}/{len(chunks)}]\n{ch}",
            questions=qs,
            context_note=context_note + " Answer only if the information is present in this chunk.",
            temperature=0.1,
        )
        for q in qs:
            new_v = partial.get(q)
            old_v = merged.get(q)
            if new_v and (not old_v or len(new_v.strip()) > len(old_v.strip())):
                merged[q] = new_v

    return merged


def answer_questions_json_chunked_evidence(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    max_chars: int = 12000,
    overlap: int = 800,
) -> Dict[str, Dict[str, Any]]:
    qs = [q.strip() for q in questions if q and q.strip()]
    if not qs:
        return {}

    merged: Dict[str, Dict[str, Any]] = {
        q: {
            "value": None,
            "confidence": None,
            "confidence_note": None,
            "excerpt_location": {"page": None, "sentence": None},
        }
        for q in qs
    }

    chunks = _chunk_text(document_text, max_chars=max_chars, overlap=overlap)

    for idx, ch in enumerate(chunks, start=1):
        _debug_print("PROCESSING CHUNK (evidence)", f"{idx}/{len(chunks)}")

        partial = answer_questions_json_evidence(
            model=model,
            document_text=ch,
            questions=qs,
            context_note=(
                context_note
                + f" This is chunk {idx}/{len(chunks)}. Answer only if the information is present in this chunk."
            ),
            temperature=0.1,
        )

        for q in qs:
            old_item = merged.get(q, {})
            new_item = partial.get(q, {})
            if _rich_answer_score(new_item) > _rich_answer_score(old_item):
                merged[q] = new_item

    return merged