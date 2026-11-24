"""Text Qwen3 SFT: response-only labels, inference-identical non-thinking prefix.

Run --validate-only against a local tokenizer before allocating GPU memory.
No data download, remote log export, deployment or service replacement is automatic.
"""
import argparse
import hashlib
import json
from pathlib import Path


def tokenize_example(row, tokenizer, max_length):
    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    completion = row["completion"]
    if len(completion) != 1 or completion[0]["role"] != "assistant":
        raise ValueError("expected exactly one assistant completion")
    label = completion[0]["content"]
    if label not in {"triage", "preconsult", "info", "other"}:
        raise ValueError("target must be one of four labels, without rationale")
    end = "<|im_end|>"
    if tokenizer.convert_tokens_to_ids(end) in {None, tokenizer.unk_token_id}:
        raise ValueError("Qwen end-of-message token missing")
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + label + end, add_special_tokens=False)
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("template prefix token boundary mismatch")
    if len(full_ids) > max_length:
        # Do not silently remove the latest user goal or the supervised label.
        raise ValueError(f"sequence length {len(full_ids)} exceeds {max_length}; shorten context or increase --max-length")
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    if not any(t != -100 for t in labels):
        raise ValueError("no target tokens after masking")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output", default="checkpoints/qwen3-8b-intent")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gradient-accumulation", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261118)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    if config.model_type != "qwen3":
        raise ValueError(f"requires text Qwen3, found {config.model_type}; VL weights are not interchangeable")
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    def load(path):
        values = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
        if not values:
            raise ValueError("empty dataset")
        return [tokenize_example(r, tokenizer, a.max_length) for r in values]
    train, dev = load(a.train), load(a.dev)
    print(json.dumps({"train": len(train), "dev": len(dev),
                      "max_tokens": max(len(r['input_ids']) for r in train + dev),
                      "first_target": tokenizer.decode([x for x in train[0]['labels'] if x != -100]),
                      "thinking_enabled": False}, ensure_ascii=False), flush=True)
    if a.validate_only:
        return
    import platform
    import torch
    import transformers
    import peft
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA configuration")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported; select a verified GPU/software stack")
    set_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=a.local_files_only, torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=a.rank, lora_alpha=2*a.rank,
                                          lora_dropout=.05, target_modules=["q_proj", "v_proj"]))
    model.print_trainable_parameters()
    class Collator:
        def __call__(self, features):
            length = max(len(r["input_ids"]) for r in features)
            result = {}
            for key, pad in [("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)]:
                result[key] = torch.tensor([r[key] + [pad]*(length-len(r[key])) for r in features], dtype=torch.long)
            return result
    args = TrainingArguments(output_dir=a.output, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.gradient_accumulation, learning_rate=a.learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=.03, weight_decay=.01,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", logging_steps=10, save_strategy="epoch", eval_strategy="epoch",
        save_total_limit=3, report_to="none", seed=a.seed, data_seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(train),
                      eval_dataset=Dataset.from_list(dev), data_collator=Collator(), processing_class=tokenizer)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"arguments": vars(a), "model_commit": getattr(config, "_commit_hash", None),
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "transformers": transformers.__version__, "peft": peft.__version__},
                "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                "data_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                for name, path in [("train", a.train), ("dev", a.dev)]}}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer.train()
    trainer.save_model(a.output)
    tokenizer.save_pretrained(a.output)


if __name__ == "__main__":
    main()
