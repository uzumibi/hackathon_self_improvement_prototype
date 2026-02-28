import asyncio
import json
import os
from pathlib import Path

import weave
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

DATA_PATH = Path("./data")
DATA_FILE = DATA_PATH / "music_request_dataset.jsonl"
PROJECT_NAME = "music-params-eval"
MODEL_NAME = os.getenv("EVAL_MODEL", "open-mistral-7b")
NUM_SAMPLES = int(os.getenv("EVAL_NUM_SAMPLES", "100"))
TOLERANCE = float(os.getenv("EVAL_TOLERANCE", "0.2"))
TOP_K_FAILURES = int(os.getenv("EVAL_TOP_K_FAILURES", "3"))
FAILURE_REPORT_PATH = Path(
    os.getenv("EVAL_FAILURE_REPORT", str(DATA_PATH / "eval_failure_report.json"))
)

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

要望文:
{request}
"""

AXES = [
    "energy",
    "warmth",
    "brightness",
    "acousticness",
    "complexity",
    "nostalgia",
]


def clip(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_eval_dataset(rows):
    dataset = []
    for row in rows:
        dataset.append(
            {
                "request": row["request"],
                "target": row["parameters"],
            }
        )
    return dataset


client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
weave.init(PROJECT_NAME)


@weave.op()
def call_mistral(model: str, messages: list, **kwargs):
    response = client.chat(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    raw = json.loads(response.choices[0].message.content)
    normalized = {}
    for axis in AXES:
        value = raw.get(axis, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        normalized[axis] = round(clip(value), 3)
    return normalized


class MistralParamModel(weave.Model):
    model: str
    prompt: str
    temperature: float = 0.0

    @weave.op
    def create_messages(self, request: str):
        return [
            ChatMessage(
                role="user",
                content=self.prompt.format(request=request),
            )
        ]

    @weave.op
    def predict(self, request: str):
        messages = self.create_messages(request)
        return call_mistral(self.model, messages, temperature=self.temperature)


def evaluate_examples(model: MistralParamModel, eval_ds, tolerance: float, top_k: int):
    if not eval_ds:
        return {
            "overall_mae": None,
            "within_tolerance_rate": 0.0,
            "mae_per_axis": {axis: None for axis in AXES},
            "failure_counts_per_axis": {axis: 0 for axis in AXES},
            "axis_failure_examples": {axis: [] for axis in AXES},
        }

    axis_error_sums = {axis: 0.0 for axis in AXES}
    axis_failures = {axis: [] for axis in AXES}
    within_tolerance_count = 0

    for row in eval_ds:
        request = row["request"]
        target = row["target"]
        prediction = model.predict(request)

        abs_errors = {}
        for axis in AXES:
            pred_val = float(prediction.get(axis, 0.0))
            target_val = float(target.get(axis, 0.0))
            err = abs(pred_val - target_val)
            abs_errors[axis] = err
            axis_error_sums[axis] += err

            if err > tolerance:
                axis_failures[axis].append(
                    {
                        "request": request,
                        "target": target_val,
                        "prediction": pred_val,
                        "abs_error": round(err, 4),
                    }
                )

        if all(abs_errors[axis] <= tolerance for axis in AXES):
            within_tolerance_count += 1

    n = len(eval_ds)
    mae_per_axis = {axis: axis_error_sums[axis] / n for axis in AXES}
    overall_mae = sum(mae_per_axis.values()) / len(AXES)
    within_tolerance_rate = within_tolerance_count / n

    axis_failure_examples = {}
    failure_counts_per_axis = {}
    for axis in AXES:
        sorted_failures = sorted(
            axis_failures[axis], key=lambda x: x["abs_error"], reverse=True
        )
        axis_failure_examples[axis] = sorted_failures[:top_k]
        failure_counts_per_axis[axis] = len(axis_failures[axis])

    return {
        "overall_mae": overall_mae,
        "within_tolerance_rate": within_tolerance_rate,
        "mae_per_axis": mae_per_axis,
        "failure_counts_per_axis": failure_counts_per_axis,
        "axis_failure_examples": axis_failure_examples,
    }


async def main():
    rows = read_jsonl(DATA_FILE)
    eval_rows = rows[:NUM_SAMPLES]
    eval_ds = build_eval_dataset(eval_rows)

    model = MistralParamModel(model=MODEL_NAME, prompt=PROMPT, temperature=0.0)
    result = evaluate_examples(model, eval_ds, tolerance=TOLERANCE, top_k=TOP_K_FAILURES)

    FAILURE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    failure_report = {
        "model": MODEL_NAME,
        "num_samples": len(eval_ds),
        "tolerance": TOLERANCE,
        "failure_counts_per_axis": result["failure_counts_per_axis"],
        "axis_failure_examples": result["axis_failure_examples"],
    }
    with open(FAILURE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(failure_report, f, ensure_ascii=False, indent=2)

    output = {
        "overall_mae": result["overall_mae"],
        "within_tolerance_rate": result["within_tolerance_rate"],
        "mae_per_axis": result["mae_per_axis"],
        "failure_counts_per_axis": result["failure_counts_per_axis"],
        "failure_report_path": str(FAILURE_REPORT_PATH),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
