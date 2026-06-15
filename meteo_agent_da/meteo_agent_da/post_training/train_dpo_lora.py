from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_preference_rows(path: Path) -> List[Dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(
            {
                "prompt": _messages_to_text(raw.get("prompt", [])),
                "chosen": _messages_to_text(raw.get("chosen", [])),
                "rejected": _messages_to_text(raw.get("rejected", [])),
            }
        )
    return rows


def write_training_plan(args: argparse.Namespace, sample_count: int) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "stage": "dpo_lora",
        "dry_run": not args.execute,
        "model": args.model,
        "preference_file": args.preference_file,
        "sample_count": sample_count,
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "beta": args.beta,
        "note": "Use --execute only when trl, transformers, datasets, and peft are installed.",
    }
    path = output_dir / "dpo_training_plan.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_execute(args: argparse.Namespace, rows: List[Dict[str, str]]) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "DPO execution requires optional dependencies: trl, transformers, datasets, peft. "
            "Run without --execute to create a dry-run training plan."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
        ),
    )
    dataset = Dataset.from_list(rows)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=max(args.max_steps, 1),
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for message in messages or []:
        parts.append(f"<|{message.get('role', '')}|>\n{message.get('content', '')}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny LoRA DPO wrapper for MeteoAgent-DA preference data.")
    parser.add_argument("--preference-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="runs/post_training/dpo_lora")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    rows = load_preference_rows(Path(args.preference_file))
    plan_path = write_training_plan(args, len(rows))
    if not args.execute:
        print(json.dumps({"dry_run": True, "samples": len(rows), "plan": str(plan_path)}, ensure_ascii=False))
        return
    run_execute(args, rows)


if __name__ == "__main__":
    main()
