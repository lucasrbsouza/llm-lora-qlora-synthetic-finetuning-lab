from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/football-lora")


@dataclass
class DatasetSettings:
    domain: str = "football"
    total_pairs: int = 60
    train_ratio: float = 0.9
    random_seed: int = 42
    train_path: Path = field(default_factory=lambda: DATA_DIR / "train.jsonl")
    test_path: Path = field(default_factory=lambda: DATA_DIR / "test.jsonl")


@dataclass
class QuantHyperparams:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = False


@dataclass
class LoraHyperparams:
    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainHyperparams:
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    optim: str = "paged_adamw_32bit"
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    fp16: bool = True
    max_seq_length: int = 512
    save_strategy: str = "no"
