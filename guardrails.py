"""
Input guardrail — TWO layers.

  Layer 1: ProtectAI deberta-v3-base-prompt-injection-v2
           A classifier trained specifically to detect prompt injection /
           jailbreak *manipulation patterns* (e.g. "ignore previous
           instructions", "SYSTEM: override safety"). This is what catches
           prompt_injection and rag_poisoning attacks.

  Layer 2: Llama Guard 3, via Ollama
           A content-safety classifier trained on harm categories (violence,
           weapons, hate speech, sexual content, etc). This is what catches
           jailbreak requests for genuinely harmful info and toxicity.

Why two models: Llama Guard alone was not built to recognize injection /
manipulation phrasing — it looks at whether the content is harmful, not
whether the instruction is trying to hijack the model. A dedicated
injection classifier is trained on exactly that pattern. Running both
covers both attack families.
"""
from transformers import pipeline
from langchain_ollama import ChatOllama
import json
import time
from pathlib import Path

# ── Decision logging ──
# Every guardrail decision (allowed or blocked) is appended as one JSON line
# to logs/guardrail_log.jsonl. This is the "did it actually work in
# production" record — without it there's no way to notice drift over time,
# or to find out a real user got wrongly blocked unless they tell you.
LOG_PATH = Path(__file__).resolve().parent / "logs" / "guardrail_log.jsonl"


def _log_decision(text: str, result: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query": text,
            "flagged": result["flagged"],
            "layer": result["layer"],
            "injection_score": result["injection_score"],
            "categories": result["categories"],
            "error": result["error"],
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Logging must never break the guardrail itself.
        print(f"  [guardrail] failed to write log entry ({e})")


# ── Layer 1: dedicated prompt-injection / jailbreak-pattern classifier ──
INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
_injection_classifier = pipeline("text-classification", model=INJECTION_MODEL)
INJECTION_THRESHOLD = 0.75   # confidence required to flag as injection

# ── Layer 2: content-safety classifier ──
GUARD_MODEL = "llama-guard3"
_guard_llm = ChatOllama(model=GUARD_MODEL, temperature=0)


def _check_injection(text: str) -> dict:
    try:
        result = _injection_classifier(text, truncation=True)[0]
        # this model's positive label is "INJECTION"; everything else is safe
        is_injection = result["label"].upper() == "INJECTION" and result["score"] >= INJECTION_THRESHOLD
        return {"flagged": is_injection, "score": round(result["score"], 3), "error": None}
    except Exception as e:
        print(f"  [guardrail] injection classifier failed ({e}) -> failing closed")
        return {"flagged": True, "score": None, "error": str(e)}


def _check_content_safety(text: str) -> dict:
    try:
        response = _guard_llm.invoke(text)
        result_text = response.content.strip().lower()
    except Exception as e:
        print(f"  [guardrail] Llama Guard call failed ({e}) -> failing closed")
        return {"flagged": True, "categories": [], "error": str(e)}

    if result_text.startswith("unsafe"):
        lines = result_text.splitlines()
        categories = [c.strip() for c in (lines[1].split(",") if len(lines) > 1 else []) if c.strip()]
        return {"flagged": True, "categories": categories, "error": None}

    return {"flagged": False, "categories": [], "error": None}


def check_input_safety(text: str) -> dict:
    """
    Run both layers and combine the result. A query is blocked if EITHER
    layer flags it.

    Returns:
        {
          "flagged": bool,
          "categories": [str, ...],   # Llama Guard hazard codes, if any
          "injection_score": float | None,
          "layer": "injection" | "content_safety" | None,  # which layer caught it
          "error": str | None
        }
    """
    injection_result = _check_injection(text)
    if injection_result["flagged"]:
        result = {
            "flagged": True,
            "categories": [],
            "injection_score": injection_result["score"],
            "layer": "injection",
            "error": injection_result["error"],
        }
        _log_decision(text, result)
        return result

    content_result = _check_content_safety(text)
    if content_result["flagged"]:
        result = {
            "flagged": True,
            "categories": content_result["categories"],
            "injection_score": injection_result["score"],
            "layer": "content_safety",
            "error": content_result["error"],
        }
        _log_decision(text, result)
        return result

    result = {
        "flagged": False,
        "categories": [],
        "injection_score": injection_result["score"],
        "layer": None,
        "error": None,
    }
    _log_decision(text, result)
    return result