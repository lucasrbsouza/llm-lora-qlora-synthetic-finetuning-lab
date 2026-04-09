import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from config import DatasetSettings, LoraHyperparams, QuantHyperparams, TrainHyperparams


def load_env() -> dict:
    load_dotenv()
    return {
        "hf_token": os.environ.get("HF_TOKEN"),
        "base_model": os.environ.get("BASE_MODEL", "NousResearch/Llama-2-7b-hf"),
    }


def build_bnb_config(cfg: QuantHyperparams) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def load_model_and_tokenizer(model_name: str, bnb_config: BitsAndBytesConfig, hf_token: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_lora_config(cfg: LoraHyperparams) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
        target_modules=cfg.target_modules,
    )


def apply_lora(model, lora_cfg: LoraConfig):
    return get_peft_model(model, lora_cfg)


def load_datasets(cfg: DatasetSettings):
    data_files = {
        "train": str(cfg.train_path),
        "test": str(cfg.test_path),
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset["train"], dataset["test"]


def format_prompt(example: dict) -> dict:
    text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    return {"text": text}


def build_training_args(cfg: TrainHyperparams) -> TrainingArguments:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        optim=cfg.optim,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        fp16=cfg.fp16,
        save_strategy=cfg.save_strategy,
        report_to="none",
    )


def build_trainer(model, tokenizer, train_ds, eval_ds, training_args, cfg: TrainHyperparams) -> SFTTrainer:
    return SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
    )


def main():
    env = load_env()

    quant_cfg = QuantHyperparams()
    lora_cfg = LoraHyperparams()
    train_cfg = TrainHyperparams()
    data_cfg = DatasetSettings()

    bnb_config = build_bnb_config(quant_cfg)
    model, tokenizer = load_model_and_tokenizer(env["base_model"], bnb_config, env["hf_token"])

    lora_config = build_lora_config(lora_cfg)
    model = apply_lora(model, lora_config)

    train_ds, eval_ds = load_datasets(data_cfg)
    train_ds = train_ds.map(format_prompt)
    eval_ds = eval_ds.map(format_prompt)

    training_args = build_training_args(train_cfg)
    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, training_args, train_cfg)

    print("Starting training...")
    trainer.train()

    trainer.model.save_pretrained(str(train_cfg.output_dir))
    tokenizer.save_pretrained(str(train_cfg.output_dir))
    print(f"Adapter saved to {train_cfg.output_dir}")


if __name__ == "__main__":
    main()
