import json
import os
from pathlib import Path

from mistralai.client import MistralClient
from mistralai.models.jobs import TrainingParameters, WandbIntegrationIn

DATA_PATH = Path("./data")
RAW_DATA_FILE = DATA_PATH / "music_request_dataset.jsonl"
TRAIN_FILE = DATA_PATH / "music_request_train_formatted.jsonl"
VAL_FILE = DATA_PATH / "music_request_val_formatted.jsonl"

PROJECT_NAME = "music-params-ft"
BASE_MODEL = os.getenv("FT_BASE_MODEL", "mistral-3b-latest")
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.9"))
TRAINING_STEPS = int(os.getenv("TRAINING_STEPS", "35"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))

PROMPT = """あなたは音楽の印象を6軸パラメータで推定するアシスタントです。
入力された要望文から、以下の6つの値をそれぞれ -1.0 〜 1.0 の範囲で推定してください。

- energy: 静寂(-1.0) ↔ 熱狂(1.0)
- warmth: 無機質(-1.0) ↔ 温もり(1.0)
- brightness: 暗闇(-1.0) ↔ 晴天(1.0)
- acousticness: 電子音(-1.0) ↔ 生楽器(1.0)
- complexity: 単純(-1.0) ↔ 複雑(1.0)
- nostalgia: 未来的(-1.0) ↔ 懐古的(1.0)

必ずJSONのみを返してください。キーは次の6つ固定です:
energy, warmth, brightness, acousticness, complexity, nostalgia
"""


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(rows, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_example(row, with_answer=True):
    request_text = row["request"]
    params = row["parameters"]

    messages = [
        {
            "role": "user",
            "content": (
                f"{PROMPT}\n\n"
                f"要望文:\n{request_text}\n\n"
                "JSONで回答してください。"
            ),
        }
    ]

    if with_answer:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(params, ensure_ascii=False),
            }
        )

    return messages


def split_dataset(rows, train_ratio=0.9):
    n_total = len(rows)
    n_train = max(1, int(n_total * train_ratio))
    n_train = min(n_train, n_total - 1) if n_total > 1 else n_total
    return rows[:n_train], rows[n_train:]


def main():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set")

    raw_rows = read_jsonl(RAW_DATA_FILE)
    if len(raw_rows) < 2:
        raise ValueError("Need at least 2 rows in music_request_dataset.jsonl")

    train_rows, val_rows = split_dataset(raw_rows, TRAIN_RATIO)

    train_formatted = [format_example(row, with_answer=True) for row in train_rows]
    val_formatted = [format_example(row, with_answer=True) for row in val_rows]

    save_jsonl(train_formatted, TRAIN_FILE)
    save_jsonl(val_formatted, VAL_FILE)

    client = MistralClient(api_key=api_key)

    with open(TRAIN_FILE, "rb") as f:
        train_file = client.files.create(file=(TRAIN_FILE.name, f))
    with open(VAL_FILE, "rb") as f:
        val_file = client.files.create(file=(VAL_FILE.name, f))

    integrations = []
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        integrations.append(
            WandbIntegrationIn(project=PROJECT_NAME, api_key=wandb_api_key).dict()
        )

    job = client.jobs.create(
        model=BASE_MODEL,
        training_files=[train_file.id],
        validation_files=[val_file.id],
        hyperparameters=TrainingParameters(
            training_steps=TRAINING_STEPS,
            learning_rate=LEARNING_RATE,
        ),
        integrations=integrations,
    )

    print("Fine-tuning job created")
    print(f"job_id: {job.id}")


if __name__ == "__main__":
    main()
