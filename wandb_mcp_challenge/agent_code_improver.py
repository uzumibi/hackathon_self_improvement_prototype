import json
from pathlib import Path
from typing import Any, Dict

from mistralai.client import MistralClient

ROOT = Path(__file__).resolve().parents[1]
TARGET_FILE = ROOT / "wandb_mcp_challenge" / "agent_modifiable.py"

BASE_PROMPT = """あなたはMLシステムの改善を行うコーディングエージェントです。
目的は、文章から6軸パラメタを推定する性能を上げることです。

入力として与える情報:
- 評価サマリ（overall_mae, mae_per_axis, worst_axis）
- 失敗例テキスト
- 既存の軸別追加指示

あなたの出力はJSONのみ:
{
  "axis": "修正対象軸",
  "new_instruction": "その軸の曖昧性を下げる短い指示",
  "reason": "1文"
}

制約:
- new_instructionは日本語1文
- 既存指示より具体的にする
- 禁止: コードブロック、説明文、余計なキー
"""


def _safe_json_load(text: str) -> Dict[str, Any]:
    return json.loads(text)


def _write_target_module(instructions: Dict[str, str]):
    lines = ["AXIS_EXTRA_INSTRUCTIONS = {"]
    for axis, guide in instructions.items():
        escaped = guide.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    "{axis}": "{escaped}",')
    lines.append("}\n")
    TARGET_FILE.write_text("\n".join(lines), encoding="utf-8")


def improve_axis_instruction(
    api_key: str,
    eval_summary: Dict[str, Any],
    failure_report_path: str,
) -> Dict[str, Any]:
    failure_report = {}
    report_file = ROOT / failure_report_path if failure_report_path else None
    if report_file and report_file.exists():
        failure_report = json.loads(report_file.read_text(encoding="utf-8"))

    namespace: Dict[str, Any] = {}
    exec(TARGET_FILE.read_text(encoding="utf-8"), namespace)
    current_instructions = dict(namespace.get("AXIS_EXTRA_INSTRUCTIONS", {}))

    client = MistralClient(api_key=api_key)

    user_payload = {
        "eval_summary": eval_summary,
        "failure_report": failure_report,
        "current_instructions": current_instructions,
    }

    response = client.chat(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    payload = _safe_json_load(response.choices[0].message.content)
    axis = payload.get("axis")
    new_instruction = payload.get("new_instruction")

    if axis not in current_instructions or not isinstance(new_instruction, str) or not new_instruction.strip():
        return {
            "updated": False,
            "reason": "invalid agent suggestion",
            "suggestion": payload,
        }

    current_instructions[axis] = new_instruction.strip()
    _write_target_module(current_instructions)

    return {
        "updated": True,
        "axis": axis,
        "new_instruction": new_instruction.strip(),
        "reason": payload.get("reason", ""),
        "target_file": str(TARGET_FILE),
    }
