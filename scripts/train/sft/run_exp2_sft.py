#!python
"""
E2: SFT 基线训练。

在 BFCL 单轮数据上对 Qwen2.5-1.5B 做 supervised fine-tuning。
配置由 configs/exp2_sft.yaml 提供，可通过 EXP_CONFIG 环境变量覆盖。
"""

import os
import sys
import json
from pathlib import Path

# 项目路径
PROJECT_DIR = str(Path(__file__).resolve().parent.parent.parent.parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "verl"))

import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from loguru import logger


def load_config():
    cfg_path = os.environ.get("EXP_CONFIG", os.path.join(PROJECT_DIR, "configs", "exp2_sft.yaml"))
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"配置加载: {cfg_path}")
    return cfg


def main():
    cfg = load_config()
    model_path = cfg["model"]["path"]
    train_files = os.path.join(PROJECT_DIR, cfg["data"]["train_file"])
    val_files = os.path.join(PROJECT_DIR, cfg["data"]["val_file"])
    MAX_SEQ_LEN = int(cfg["data"]["max_seq_len"])
    output_dir = os.path.join(PROJECT_DIR, "checkpoints", cfg["exp"]["name"])
    tcfg = cfg["train"]

    logger.info(f"加载数据: train={train_files}, val={val_files}")
    dataset = load_dataset("parquet", data_files={
        "train": train_files,
        "validation": val_files,
    })

    logger.info(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        prompts_raw = examples["prompt"]
        targets_raw = examples.get("target", [""] * len(prompts_raw))

        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for prompt_str, target_str in zip(prompts_raw, targets_raw):
            try:
                messages = json.loads(prompt_str) if isinstance(prompt_str, str) else prompt_str
            except json.JSONDecodeError:
                messages = [{"role": "user", "content": str(prompt_str)}]

            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            target_text = target_str if target_str else ""

            full_text = prompt_text + target_text
            encoded = tokenizer(full_text, max_length=MAX_SEQ_LEN, truncation=True, return_tensors=None)
            ids = encoded["input_ids"]
            mask = encoded["attention_mask"]

            prompt_encoded = tokenizer(prompt_text, max_length=MAX_SEQ_LEN, truncation=True, return_tensors=None)
            prompt_len = len(prompt_encoded["input_ids"])

            if prompt_len >= MAX_SEQ_LEN:
                logger.warning(f"Prompt exceeds MAX_SEQ_LEN ({MAX_SEQ_LEN}), target truncated")

            labels = [-100] * min(prompt_len, len(ids)) + ids[prompt_len:]
            ids = ids[:MAX_SEQ_LEN]
            labels = labels[:MAX_SEQ_LEN]
            mask = mask[:MAX_SEQ_LEN]

            input_ids_list.append(ids)
            labels_list.append(labels)
            attention_mask_list.append(mask)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    remove_cols = [c for c in dataset["train"].column_names if c not in ("input_ids", "attention_mask", "labels")]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    logger.info(f"数据 token 化完成: train={len(tokenized['train'])}, val={len(tokenized['validation'])}")

    logger.info(f"加载模型: {model_path}")
    import torch
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    if bool(tcfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        num_train_epochs=int(tcfg["num_train_epochs"]),
        max_steps=int(os.environ.get("SFT_MAX_STEPS", "-1")),
        learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        logging_steps=int(tcfg["logging_steps"]),
        save_steps=int(tcfg["save_steps"]),
        eval_steps=int(tcfg["eval_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        bf16=bool(tcfg["bf16"]),
        dataloader_num_workers=int(tcfg["dataloader_num_workers"]),
        report_to=cfg["logging"]["report_to"],
        run_name=cfg["exp"]["name"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    logger.info("开始 SFT 训练")
    trainer.train()
    trainer.save_model()
    logger.info(f"SFT 训练完成: {output_dir}")


if __name__ == "__main__":
    main()
