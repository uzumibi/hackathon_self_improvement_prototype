import os
import json
import random
from pathlib import Path
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from wandb_mcp_challenge.agent_modifiable import AXIS_EXTRA_INSTRUCTIONS

# --- 設定 ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=MISTRAL_API_KEY)

DATA_PATH = Path("./data")
DATA_PATH.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_PATH / "music_request_dataset.jsonl"

# 生成するサンプル数
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "2"))
# 既存データに追記するかどうか
APPEND_DATASET = os.getenv("APPEND_DATASET", "0") == "1"
# 苦手軸を意識した追加生成を行う場合に指定（例: nostalgia）
FOCUS_AXIS = os.getenv("FOCUS_AXIS")
FOCUS_STRENGTH = float(os.getenv("FOCUS_STRENGTH", "0.6"))
FAILURE_HINTS_PATH = os.getenv("FAILURE_HINTS_PATH")
AGENT_POLICY_PATH = os.getenv(
    "AGENT_POLICY_PATH", "./wandb_mcp_challenge/agent_policy.json"
)
# 依頼文生成に使用するモデル（データ品質重視なら large がおすすめ）
GEN_MODEL = "mistral-large-latest" 

# --- プロンプト定義 ---
BASE_SYSTEM_PROMPT = """あなたはプロの音楽評論家兼シナリオライターです。
与えられた6つの音響パラメータ(-1.0 to 1.0)の数値を解釈し、そのニュアンスを反映した「客の自然な要望文」を1つだけ生成してください。

【パラメータ定義】
1. Energy: -1.0(静寂) ↔ 1.0(熱狂)
2. Warmth: -1.0(無機質) ↔ 1.0(温もり)
3. Brightness: -1.0(暗闇) ↔ 1.0(晴天)
4. Acousticness: -1.0(電子音) ↔ 1.0(生楽器)
5. Complexity: -1.0(単純) ↔ 1.0(複雑)
6. Nostalgia: -1.0(未来的) ↔ 1.0(懐古的)

【制約】
- 文章の中に数値は絶対に出さない。
- 情景描写、感情、比喩を用いて、そのパラメータの音楽が流れていそうなシチュエーションを描く。
- 出力はJSON形式 {"request": "文章"} のみとすること。"""

# --- 処理 ---

def load_failure_hints(path: str):
    if not path:
        return None
    hint_path = Path(path)
    if not hint_path.exists():
        print(f"Hint file not found: {hint_path}")
        return None
    try:
        with open(hint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load hint file: {e}")
        return None


def load_agent_policy(path: str):
    policy_path = Path(path)
    if not policy_path.exists():
        return None
    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load policy file: {e}")
        return None


def build_hint_block(hints):
    if not hints:
        return ""
    axis_examples = hints.get("axis_failure_examples", {})
    if not axis_examples:
        return ""

    lines = [
        "【改善ヒント（前回評価で予測誤差が大きかった要望文例）】",
        "次の失敗例を参考に、曖昧さが減るように要望文を設計してください。",
    ]
    for axis, examples in axis_examples.items():
        if not examples:
            continue
        lines.append(f"- {axis}")
        for example in examples[:2]:
            request = str(example.get("request", "")).replace("\n", " ").strip()
            if len(request) > 140:
                request = request[:140] + "..."
            lines.append(f"  例: {request}")

    return "\n".join(lines)


def build_system_prompt(hints):
    prompt = BASE_SYSTEM_PROMPT
    hint_block = build_hint_block(hints)
    if hint_block:
        prompt += "\n\n" + hint_block
    policy = load_agent_policy(AGENT_POLICY_PATH)
    if policy:
        axis_guidance = policy.get("axis_guidance", {})
        if axis_guidance:
            lines = [
                "【エージェント改善ポリシー】",
                "次の軸ガイダンスを満たすように文章を設計してください。",
            ]
            for axis, guide in axis_guidance.items():
                lines.append(f"- {axis}: {guide}")
            prompt += "\n\n" + "\n".join(lines)
    if AXIS_EXTRA_INSTRUCTIONS:
        lines = [
            "【コーディングエージェントが自動更新した追加指示】",
            "各軸について次の表現上の制約も守ってください。",
        ]
        for axis, guide in AXIS_EXTRA_INSTRUCTIONS.items():
            lines.append(f"- {axis}: {guide}")
        prompt += "\n\n" + "\n".join(lines)
    if FOCUS_AXIS:
        prompt += (
            f"\n\n【追加方針】{FOCUS_AXIS}軸の違いが読み取りやすくなるよう、"
            "その軸に対応する描写を明確に含めてください。"
        )
    return prompt

def generate_random_params():
    axes = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"]
    params = {axis: round(random.uniform(-1.0, 1.0), 1) for axis in axes}
    if FOCUS_AXIS in axes:
        direction = random.choice([-1.0, 1.0])
        params[FOCUS_AXIS] = round(direction * random.uniform(FOCUS_STRENGTH, 1.0), 1)
    return params

def main():
    samples = []
    print(f"Starting generation of {NUM_SAMPLES} samples...")
    hints = load_failure_hints(FAILURE_HINTS_PATH)
    system_prompt = build_system_prompt(hints)

    for i in range(NUM_SAMPLES):
        # 1. 隠れパラメータをランダム生成
        params = generate_random_params()
        
        # 2. 上位モデルに要望文を書かせる
        content = f"Parameters: {json.dumps(params)}"
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=content)
        ]
        
        try:
            response = client.chat(
                model=GEN_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            res_json = json.loads(response.choices[0].message.content)
            
            # 3. ペアデータを保存
            data_pair = {
                "parameters": params,
                "request": res_json["request"]
            }
            samples.append(data_pair)

        except Exception as e:
            print(f"Error at sample {i}: {e}")

    # JSONL形式で書き出し
    write_mode = "a" if APPEND_DATASET and OUTPUT_FILE.exists() else "w"
    with open(OUTPUT_FILE, write_mode, encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nSuccess! Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()