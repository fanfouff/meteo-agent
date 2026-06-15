from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_sft_texts(path: Path) -> List[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parts = []
        for message in row.get("messages", []) or []:
            role = message.get("role", "")
            content = message.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        texts.append("\n".join(parts))
    return texts


def write_training_plan(args: argparse.Namespace, sample_count: int) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "stage": "sft_lora",
        "dry_run": not args.execute,
        "model": args.model,
        "train_file": args.train_file,
        "sample_count": sample_count,
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "note": "Use --execute only with a local model path and installed transformers/peft/datasets.",
    }
    path = output_dir / "sft_training_plan.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_execute(args: argparse.Namespace, texts: List[str]) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "SFT execution requires optional dependencies: transformers, datasets, peft. "
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

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny LoRA SFT wrapper for MeteoAgent-DA trajectory data.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="runs/post_training/sft_lora")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    texts = load_sft_texts(Path(args.train_file))
    plan_path = write_training_plan(args, len(texts))
    if not args.execute:
        print(json.dumps({"dry_run": True, "samples": len(texts), "plan": str(plan_path)}, ensure_ascii=False))
        return
    run_execute(args, texts)


if __name__ == "__main__":
    main()
