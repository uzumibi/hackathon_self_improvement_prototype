import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mistralai.client import MistralClient

try:
    import wandb
except Exception:
    wandb = None

from wandb_mcp_challenge.wandb_mcp_client import WandbMCPClient

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "wandb_mcp_challenge" / "runs"
GEN_SKILLS_DIR = ROOT / "wandb_mcp_challenge" / "generated_skills"
POLICY_PATH = ROOT / "wandb_mcp_challenge" / "agent_policy.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
GEN_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

DATA_SCRIPT = "data_generation.py"
FT_SCRIPT = "ft_music_params.py"
EVAL_SCRIPT = "eval_music_params.py"
RAW_DATA_FILE = ROOT / "data" / "music_request_dataset.jsonl"

MAX_ITERS = int(os.getenv("MCP_LOOP_MAX_ITERS", "3"))
POLL_INTERVAL_SEC = int(os.getenv("MCP_LOOP_POLL_INTERVAL", "30"))
POLL_TIMEOUT_SEC = int(os.getenv("MCP_LOOP_POLL_TIMEOUT", "7200"))
TARGET_MAE = float(os.getenv("MCP_LOOP_TARGET_MAE", "0.20"))
MAE_EPS = float(os.getenv("MCP_LOOP_MAE_EPS", "0.01"))
EVAL_SAMPLES = int(os.getenv("MCP_LOOP_EVAL_SAMPLES", "100"))
SKIP_INITIAL_GENERATION_IF_DATA = os.getenv("MCP_LOOP_SKIP_INITIAL_GENERATION_IF_DATA", "1") == "1"
ALLOW_EVAL_FALLBACK_ON_FT_FAILURE = os.getenv("MCP_LOOP_ALLOW_EVAL_FALLBACK_ON_FT_FAILURE", "1") == "1"
EVAL_FALLBACK_MODEL = os.getenv("MCP_LOOP_EVAL_FALLBACK_MODEL", "mistral-small-latest")


@dataclass
class LoopState:
    num_samples: int = int(os.getenv("MCP_LOOP_INIT_NUM_SAMPLES", "8"))
    training_steps: int = int(os.getenv("MCP_LOOP_INIT_STEPS", "5"))
    learning_rate: float = float(os.getenv("MCP_LOOP_INIT_LR", "1e-4"))
    focus_axis: Optional[str] = None
    failure_report_path: Optional[str] = None
    policy_path: str = str(POLICY_PATH)


class LocalRunner:
    @staticmethod
    def run(script_name: str, env_overrides: Optional[Dict[str, str]] = None):
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        cmd = ["python3", script_name]
        print(f"[run] {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result


class WandbTracker:
    def __init__(self):
        self.run = None

    def start(self):
        if wandb is None:
            print("[wandb] wandb package not available, skip tracking")
            return
        project = os.getenv("WANDB_PROJECT", "music-params-mcp-loop")
        entity = os.getenv("WANDB_ENTITY")
        self.run = wandb.init(project=project, entity=entity, job_type="self-improvement-loop")

    def log_iteration(
        self,
        iteration: int,
        state: LoopState,
        eval_summary: Dict[str, Any],
        best_mae: Optional[float],
        improvement: Optional[float],
    ):
        if not self.run:
            return
        payload = {
            "iteration": iteration,
            "overall_mae": eval_summary.get("overall_mae"),
            "within_tolerance_rate": eval_summary.get("within_tolerance_rate"),
            "best_mae": best_mae,
            "improvement_vs_baseline": improvement,
            "training_steps": state.training_steps,
            "learning_rate": state.learning_rate,
            "num_samples": state.num_samples,
        }
        mae_per_axis = eval_summary.get("mae_per_axis", {})
        if isinstance(mae_per_axis, dict):
            for axis, value in mae_per_axis.items():
                payload[f"mae/{axis}"] = value
        self.run.log(payload)

    def finish(self):
        if self.run:
            self.run.finish()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def auto_improve_policy(
    policy_path: Path,
    eval_summary: Dict[str, Any],
    failure_report_path: Optional[str],
    iteration: int,
):
    if not policy_path.exists():
        return {"updated": False, "reason": "policy file missing", "path": str(policy_path)}

    policy = load_json(policy_path)
    axis_guidance = policy.get("axis_guidance", {})
    if not isinstance(axis_guidance, dict):
        return {"updated": False, "reason": "invalid axis_guidance", "path": str(policy_path)}

    worst_axis = eval_summary.get("worst_axis")
    if not worst_axis or worst_axis not in axis_guidance:
        return {"updated": False, "reason": "no worst axis", "path": str(policy_path)}

    extra_hint = None
    if failure_report_path:
        report_file = ROOT / failure_report_path
        if report_file.exists():
            report = load_json(report_file)
            examples = report.get("axis_failure_examples", {}).get(worst_axis, [])
            if examples:
                req = str(examples[0].get("request", "")).replace("\n", " ").strip()
                if len(req) > 100:
                    req = req[:100] + "..."
                extra_hint = f"失敗例を反映し曖昧さを避ける（例: {req}）"

    current = axis_guidance.get(worst_axis, "")
    addition = "極性が分かる対比表現を必ず1つ含める。"
    if extra_hint:
        addition += f" {extra_hint}"

    if addition not in current:
        axis_guidance[worst_axis] = f"{current} {addition}".strip()

    policy["axis_guidance"] = axis_guidance
    policy["revision"] = int(policy.get("revision", 1)) + 1
    policy["last_updated_axis"] = worst_axis

    save_json(policy_path, policy)

    snapshot_path = GEN_SKILLS_DIR / f"policy_iter_{iteration:02d}.json"
    save_json(snapshot_path, policy)
    return {
        "updated": True,
        "worst_axis": worst_axis,
        "path": str(policy_path),
        "snapshot": str(snapshot_path),
    }


class SkillEngine:
    @staticmethod
    def find_key_recursively(obj: Any, key: str):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for value in obj.values():
                found = SkillEngine.find_key_recursively(value, key)
                if found is not None:
                    return found
        if isinstance(obj, list):
            for item in obj:
                found = SkillEngine.find_key_recursively(item, key)
                if found is not None:
                    return found
        return None

    @staticmethod
    def parse_eval_json_obj(text: str):
        decoder = json.JSONDecoder()
        candidates = []
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    score = 0
                    if "overall_mae" in obj:
                        score += 2
                    if "mae_per_axis" in obj:
                        score += 2
                    if "within_tolerance_rate" in obj:
                        score += 1
                    if score > 0:
                        candidates.append((score, obj))
            except json.JSONDecodeError:
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def extract_job_id(text: str):
        match = re.search(r"job_id:\s*([\w\-]+)", text)
        return match.group(1) if match else None

    @staticmethod
    def summarize_eval(eval_result: Dict[str, Any]):
        overall_mae = SkillEngine.find_key_recursively(eval_result, "overall_mae")
        within_tolerance_rate = SkillEngine.find_key_recursively(eval_result, "within_tolerance_rate")
        mae_per_axis = SkillEngine.find_key_recursively(eval_result, "mae_per_axis") or {}
        failure_report_path = SkillEngine.find_key_recursively(eval_result, "failure_report_path")

        worst_axis = None
        worst_value = -1.0
        if isinstance(mae_per_axis, dict):
            for axis, val in mae_per_axis.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if v > worst_value:
                    worst_axis = axis
                    worst_value = v

        try:
            overall_mae = float(overall_mae)
        except (TypeError, ValueError):
            overall_mae = None

        try:
            within_tolerance_rate = float(within_tolerance_rate)
        except (TypeError, ValueError):
            within_tolerance_rate = None

        return {
            "overall_mae": overall_mae,
            "within_tolerance_rate": within_tolerance_rate,
            "mae_per_axis": mae_per_axis,
            "worst_axis": worst_axis,
            "failure_report_path": failure_report_path,
        }

    @staticmethod
    def propose_next(state: LoopState, eval_summary: Dict[str, Any], best_mae: Optional[float]):
        next_state = LoopState(**asdict(state))
        reason = []

        mae = eval_summary.get("overall_mae")
        worst_axis = eval_summary.get("worst_axis")

        improved = mae is not None and (best_mae is None or (best_mae - mae) > MAE_EPS)

        if not improved:
            next_state.training_steps = min(next_state.training_steps + 10, 120)
            next_state.learning_rate = max(next_state.learning_rate * 0.8, 1e-5)
            reason.append("plateau -> steps+10 and lr*0.8")
        else:
            reason.append("improved -> keep hyperparameters")

        if worst_axis:
            next_state.focus_axis = worst_axis
            next_state.num_samples = min(next_state.num_samples + 4, 40)
            reason.append(f"target data on weak axis: {worst_axis}")

        if eval_summary.get("failure_report_path"):
            next_state.failure_report_path = eval_summary["failure_report_path"]

        return next_state, " / ".join(reason)


def wait_for_job(client: MistralClient, job_id: str):
    start = time.time()
    while True:
        job = client.jobs.get(job_id=job_id)
        status = getattr(job, "status", None) or (job.get("status") if isinstance(job, dict) else None)
        print(f"[job] {job_id} status={status}")

        if status in {"SUCCESS", "succeeded", "completed"}:
            return job
        if status in {"FAILED", "failed", "canceled", "cancelled"}:
            raise RuntimeError(f"job failed: {status}")
        if time.time() - start > POLL_TIMEOUT_SEC:
            raise TimeoutError(f"timeout for job: {job_id}")

        time.sleep(POLL_INTERVAL_SEC)


def get_ft_model(job: Any):
    if isinstance(job, dict):
        return job.get("fine_tuned_model")
    return getattr(job, "fine_tuned_model", None)


def save_generated_skill_artifacts(iteration: int, payload: Dict[str, Any]):
    out = GEN_SKILLS_DIR / f"iter_{iteration:02d}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def build_submission_summary(history: list):
    if not history:
        return {"summary": "no iterations executed"}

    maes = [
        h.get("eval_summary", {}).get("overall_mae")
        for h in history
        if h.get("eval_summary", {}).get("overall_mae") is not None
    ]
    baseline_mae = maes[0] if maes else None
    best_mae = min(maes) if maes else None
    improvement = (baseline_mae - best_mae) if (baseline_mae is not None and best_mae is not None) else None

    generated_files = sorted([str(p) for p in GEN_SKILLS_DIR.glob("iter_*.json")])
    policy_snaps = sorted([str(p) for p in GEN_SKILLS_DIR.glob("policy_iter_*.json")])

    return {
        "proven_improvement": {
            "baseline_mae": baseline_mae,
            "best_mae": best_mae,
            "absolute_gain": improvement,
            "relative_gain_percent": (improvement / baseline_mae * 100.0) if (improvement and baseline_mae) else None,
        },
        "generated_skills_submitted": {
            "iteration_skill_files": generated_files,
            "policy_snapshots": policy_snaps,
            "skill_templates": [
                "wandb_mcp_challenge/skills/skill_eval_and_diagnose.md",
                "wandb_mcp_challenge/skills/skill_propose_next_action.md",
                "wandb_mcp_challenge/skills/skill_generate_targeted_data.md",
            ],
        },
    }


def main():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is required")

    state = LoopState()
    best_mae = None
    baseline_mae = None
    history = []

    tracker = WandbTracker()
    tracker.start()

    mcp_client = WandbMCPClient()
    client = MistralClient(api_key=api_key)
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT", "music-params-mcp-loop")

    try:
        for iteration in range(1, MAX_ITERS + 1):
            print(f"\n=== Iteration {iteration}/{MAX_ITERS} ===")
            print(f"[state] {asdict(state)}")

            gen_env = {
                "NUM_SAMPLES": str(state.num_samples),
                "APPEND_DATASET": "1" if iteration > 1 else "0",
                "AGENT_POLICY_PATH": state.policy_path,
            }
            if state.focus_axis:
                gen_env["FOCUS_AXIS"] = state.focus_axis
                gen_env["FOCUS_STRENGTH"] = "0.7"
            if state.failure_report_path:
                gen_env["FAILURE_HINTS_PATH"] = state.failure_report_path

            existing_rows = count_jsonl_rows(RAW_DATA_FILE)
            skip_initial_generation = (
                iteration == 1
                and SKIP_INITIAL_GENERATION_IF_DATA
                and existing_rows >= 2
            )

            if skip_initial_generation:
                print(f"[data] skip initial generation: existing rows={existing_rows}")
            else:
                gen_res = LocalRunner.run(DATA_SCRIPT, gen_env)
                if gen_res.returncode != 0:
                    raise RuntimeError("data generation failed")

            ft_env = {
                "TRAINING_STEPS": str(state.training_steps),
                "LEARNING_RATE": str(state.learning_rate),
            }
            ft_res = LocalRunner.run(FT_SCRIPT, ft_env)

            job_id = None
            ft_mode = "fine_tuned"
            if ft_res.returncode != 0:
                if not ALLOW_EVAL_FALLBACK_ON_FT_FAILURE:
                    raise RuntimeError("fine-tuning failed")
                ft_mode = "fallback_base_model"
                ft_model = EVAL_FALLBACK_MODEL
                print(f"[ft] fine-tuning failed -> fallback evaluation model: {ft_model}")
            else:
                job_id = SkillEngine.extract_job_id(ft_res.stdout + "\n" + ft_res.stderr)
                if not job_id:
                    raise RuntimeError("cannot parse job_id")
                job = wait_for_job(client, job_id)
                ft_model = get_ft_model(job)
                if not ft_model:
                    raise RuntimeError("fine-tuned model missing")

            failure_report_path = f"data/eval_failure_report_iter_{iteration}.json"
            eval_env = {
                "EVAL_MODEL": ft_model,
                "EVAL_NUM_SAMPLES": str(EVAL_SAMPLES),
                "EVAL_FAILURE_REPORT": failure_report_path,
            }
            eval_res = LocalRunner.run(EVAL_SCRIPT, eval_env)
            if eval_res.returncode != 0:
                raise RuntimeError("evaluation failed")

            eval_raw = (eval_res.stdout or "") + "\n" + (eval_res.stderr or "")
            eval_json = SkillEngine.parse_eval_json_obj(eval_raw)
            if not eval_json:
                raise RuntimeError("could not parse evaluation json")

            eval_summary = SkillEngine.summarize_eval(eval_json)
            mae = eval_summary.get("overall_mae")

            if baseline_mae is None and mae is not None:
                baseline_mae = mae
            if mae is not None and (best_mae is None or mae < best_mae):
                best_mae = mae

            next_state, reason = SkillEngine.propose_next(state, eval_summary, best_mae)
            policy_update = auto_improve_policy(
                policy_path=Path(state.policy_path),
                eval_summary=eval_summary,
                failure_report_path=eval_summary.get("failure_report_path"),
                iteration=iteration,
            )
            next_state.policy_path = state.policy_path

            mcp_snapshot = mcp_client.safe_fetch_latest_metrics(
                entity=entity,
                project=project,
                loop_context={
                    "iteration": iteration,
                    "overall_mae": eval_summary.get("overall_mae"),
                    "worst_axis": eval_summary.get("worst_axis"),
                },
            )

            skill_payload = {
                "iteration": iteration,
                "skill_eval_and_diagnose": eval_summary,
                "skill_propose_next_action": {
                    "current_state": asdict(state),
                    "next_state": asdict(next_state),
                    "reason": reason,
                },
                "skill_generate_targeted_data": {
                    "focus_axis": next_state.focus_axis,
                    "failure_report_path": next_state.failure_report_path,
                    "num_samples": next_state.num_samples,
                },
                "skill_auto_policy_improvement": policy_update,
                "skill_wandb_mcp_inspection": mcp_snapshot,
            }
            skill_artifact = save_generated_skill_artifacts(iteration, skill_payload)

            improvement = None
            if baseline_mae is not None and mae is not None:
                improvement = baseline_mae - mae

            history.append(
                {
                    "iteration": iteration,
                    "job_id": job_id,
                    "training_mode": ft_mode,
                    "eval_model": ft_model,
                    "eval_summary": eval_summary,
                    "best_mae": best_mae,
                    "improvement_vs_baseline": improvement,
                    "transition_reason": reason,
                    "policy_update": policy_update,
                    "mcp_snapshot": mcp_snapshot,
                    "generated_skill_artifact": str(skill_artifact),
                }
            )

            tracker.log_iteration(iteration, state, eval_summary, best_mae, improvement)
            print(f"[metrics] mae={eval_summary.get('overall_mae')} worst_axis={eval_summary.get('worst_axis')}")
            print(f"[decision] {reason}")
            print(f"[policy] {policy_update}")
            print(f"[artifact] {skill_artifact}")

            state = next_state

            if best_mae is not None and best_mae <= TARGET_MAE:
                print(f"[stop] target reached: {best_mae} <= {TARGET_MAE}")
                break
    finally:
        tracker.finish()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUT_DIR / f"challenge_report_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    submission_summary = build_submission_summary(history)
    submission_path = OUT_DIR / f"challenge_submission_{ts}.json"
    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(submission_summary, f, ensure_ascii=False, indent=2)

    latest_path = OUT_DIR / "challenge_submission_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(submission_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[done] report saved: {report_path}")
    print(f"[done] submission summary: {submission_path}")
    print(f"[done] latest summary: {latest_path}")


if __name__ == "__main__":
    main()