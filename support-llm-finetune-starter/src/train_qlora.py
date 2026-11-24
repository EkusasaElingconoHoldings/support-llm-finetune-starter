import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for customer support LLM")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base Hugging Face instruct model ID",
    )
    parser.add_argument("--train-file", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--eval-file", type=str, default="data/processed/holdout_eval.jsonl")
    parser.add_argument("--output-dir", type=str, default="artifacts/support-qlora-checkpoints")
    parser.add_argument("--merged-dir", type=str, default="artifacts/support-llm-merged")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-and-save", action="store_true", help="Merge LoRA adapter into base weights for serving")
    return parser.parse_args()


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    compute_dtype = get_compute_dtype()
    use_bf16 = compute_dtype == torch.bfloat16
    use_fp16 = compute_dtype == torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_files = {"train": args.train_file}
    has_eval = Path(args.eval_file).exists()
    if has_eval:
        data_files["validation"] = args.eval_file

    raw_datasets = load_dataset("json", data_files=data_files)

    def apply_chat_template(example: dict) -> dict:
        rendered = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": rendered}

    formatted_datasets = raw_datasets.map(
        apply_chat_template,
        remove_columns=raw_datasets["train"].column_names,
        desc="Applying chat template",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_eval else "no",
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_datasets["train"],
        eval_dataset=formatted_datasets.get("validation"),
        processing_class=tokenizer,
    )

    trainer.train()
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter and tokenizer -> {adapter_dir}")

    if args.merge_and_save:
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merged_model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_dir,
            device_map="auto",
            dtype=compute_dtype,
            trust_remote_code=True,
        )
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(args.merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_dir)
        print(f"Saved merged model -> {args.merged_dir}")


if __name__ == "__main__":
    main()
