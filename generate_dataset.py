import json
import os
import random

from dotenv import load_dotenv
from openai import OpenAI

from config import DATA_DIR, DatasetSettings

FOOTBALL_CATEGORIES = [
    "rules and regulations of football",
    "history and legendary players of football",
    "tactics and formations in football",
    "major competitions and tournaments in football",
    "records and statistics in football",
    "football culture and fan traditions",
]


def load_api_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]
    return OpenAI(api_key=api_key)


def build_generation_prompt(category: str, n: int) -> str:
    return (
        f"Generate {n} question-and-answer pairs about {category}. "
        "Return ONLY a valid JSON array where each element has exactly two keys: "
        '"prompt" (the question or instruction) and "response" (the answer, 2 to 4 sentences, '
        "factually accurate and educational). Do not include any text outside the JSON array."
    )


def generate_pairs_for_category(client: OpenAI, category: str, n: int) -> list[dict]:
    message = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": build_generation_prompt(category, n)}],
        temperature=0.7,
    )
    content = message.choices[0].message.content.strip()
    return json.loads(content)


def generate_all_pairs(client: OpenAI, cfg: DatasetSettings) -> list[dict]:
    pairs_per_category = cfg.total_pairs // len(FOOTBALL_CATEGORIES)
    all_pairs = []
    for category in FOOTBALL_CATEGORIES:
        pairs = generate_pairs_for_category(client, category, pairs_per_category)
        all_pairs.extend(pairs)
    return all_pairs


def split_dataset(pairs: list[dict], train_ratio: float, seed: int) -> tuple[list, list]:
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    split_index = int(len(shuffled) * train_ratio)
    return shuffled[:split_index], shuffled[split_index:]


def write_jsonl(pairs: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def main():
    cfg = DatasetSettings()
    client = load_api_client()

    print(f"Generating {cfg.total_pairs} football Q&A pairs...")
    all_pairs = generate_all_pairs(client, cfg)

    train_pairs, test_pairs = split_dataset(all_pairs, cfg.train_ratio, cfg.random_seed)

    write_jsonl(train_pairs, cfg.train_path)
    write_jsonl(test_pairs, cfg.test_path)

    print(f"Train: {len(train_pairs)} pairs -> {cfg.train_path}")
    print(f"Test:  {len(test_pairs)} pairs -> {cfg.test_path}")


if __name__ == "__main__":
    main()
