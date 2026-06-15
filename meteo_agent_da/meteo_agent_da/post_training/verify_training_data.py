from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def verify_jsonl(path: Path, data_format: str = "auto", max_content_chars: int = 80_000) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    malformed = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            malformed.append({"line": line_number, "error": str(exc)})

    missing_fields = []
    duplicate_prompts = 0
    prompts = set()
    long_samples = []
    detected_format = data_format
    if data_format == "auto":
        detected_format = _detect_format(rows)

    for index, row in enumerate(rows):
        if detected_format == "sft":
            prompt = _verify_sft_row(index, row, missing_fields)
        elif detected_format == "preference":
            prompt = _verify_preference_row(index, row, missing_fields)
        elif detected_format == "rollout_reward":
            prompt = _verify_rollout_reward_row(index, row, missing_fields)
        else:
            prompt = ""
            missing_fields.append({"index": index, "field": "known_format"})
        if prompt:
            if prompt in prompts:
                duplicate_prompts += 1
            prompts.add(prompt)
        content_size = len(json.dumps(row, ensure_ascii=False))
        if content_size > max_content_chars:
            long_samples.append({"index": index, "chars": content_size})

    valid = not malformed and not missing_fields and not long_samples
    return {
        "path": str(path),
        "format": detected_format,
        "num_rows": len(rows),
        "valid": valid,
        "malformed_count": len(malformed),
        "missing_field_count": len(missing_fields),
        "duplicate_prompt_count": duplicate_prompts,
        "long_sample_count": len(long_samples),
        "malformed": malformed[:20],
        "missing_fields": missing_fields[:50],
        "long_samples": long_samples[:20],
    }


def _detect_format(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    first = rows[0]
    if "messages" in first:
        return "sft"
    if "prompt" in first and "chosen" in first and "rejected" in first:
        return "preference"
    if "rollout" in first and "reward" in first:
        return "rollout_reward"
    return "unknown"


def _verify_sft_row(index: int, row: Dict[str, Any], missing_fields: List[Dict[str, Any]]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        missing_fields.append({"index": index, "field": "messages"})
        return ""
    roles = [item.get("role") for item in messages if isinstance(item, dict)]
    for role in ("system", "user", "assistant"):
        if role not in roles:
            missing_fields.append({"index": index, "field": f"messages.{role}"})
    assistant = next((item for item in messages if item.get("role") == "assistant"), {})
    if not str(assistant.get("content", "")).strip():
        missing_fields.append({"index": index, "field": "assistant.content"})
    user = next((item for item in messages if item.get("role") == "user"), {})
    return str(user.get("content", ""))


def _verify_preference_row(index: int, row: Dict[str, Any], missing_fields: List[Dict[str, Any]]) -> str:
    for field in ("prompt", "chosen", "rejected"):
        if not isinstance(row.get(field), list) or not row.get(field):
            missing_fields.append({"index": index, "field": field})
    prompt_messages = row.get("prompt", []) or []
    user = next((item for item in prompt_messages if isinstance(item, dict) and item.get("role") == "user"), {})
    return str(user.get("content", ""))


def _verify_rollout_reward_row(index: int, row: Dict[str, Any], missing_fields: List[Dict[str, Any]]) -> str:
    if not row.get("prompt"):
        missing_fields.append({"index": index, "field": "prompt"})
    reward = row.get("reward", {})
    if not isinstance(reward, dict) or "total_reward" not in reward:
        missing_fields.append({"index": index, "field": "reward.total_reward"})
    return str(row.get("prompt", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SFT, preference, or rollout-reward JSONL files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", choices=["auto", "sft", "preference", "rollout_reward"], default="auto")
    parser.add_argument("--max-content-chars", type=int, default=80_000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = verify_jsonl(Path(args.input), data_format=args.format, max_content_chars=args.max_content_chars)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
