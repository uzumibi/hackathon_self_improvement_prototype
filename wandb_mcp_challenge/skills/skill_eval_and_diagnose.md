# Skill: eval_and_diagnose_with_wandb_mcp (robust)

## Goal

Run evaluation, extract actionable metrics reliably, and identify weak parameter axes for the next loop.

## Inputs

- `model_name`: evaluation model identifier (fine-tuned model or fallback base model)
- `eval_script`: local eval entrypoint (default: `eval_music_params.py`)
- `num_samples`: evaluation sample count
- `iteration`: loop index

## Required output schema

```json
{
  "overall_mae": 0.0,
  "within_tolerance_rate": 0.0,
  "mae_per_axis": {
    "energy": 0.0,
    "warmth": 0.0,
    "brightness": 0.0,
    "acousticness": 0.0,
    "complexity": 0.0,
    "nostalgia": 0.0
  },
  "worst_axis": "warmth",
  "failure_report_path": "data/eval_failure_report_iter_1.json"
}
```

## Steps

1. Run local eval with:
   - `EVAL_MODEL={model_name}`
   - `EVAL_NUM_SAMPLES={num_samples}`
   - `EVAL_FAILURE_REPORT=data/eval_failure_report_iter_{iteration}.json`
2. Parse eval result from **stdout + stderr combined text**.
3. From all JSON objects in logs, select the object that best matches eval keys:
   - `overall_mae`
   - `within_tolerance_rate`
   - `mae_per_axis`
4. Determine `worst_axis` as `argmax(mae_per_axis[axis])`.
5. If W&B MCP is available, run all discovered loop tools every iteration (best-effort):
   - `query_wandb_entity_projects`: resolve accessible entity/project
   - `query_wandb_tool`: fetch run metrics such as accuracy/loss
   - `count_weave_traces_tool`: count failed traces quickly
   - `query_weave_traces_tool`: inspect failed inference steps in detail
   - `query_wandb_support_bot`: ask official bot for W&B usage guidance
   - `create_wandb_report_tool`: create before/after report for human review
6. Store MCP tool outputs as `skill_wandb_mcp_inspection` in `generated_skills/iter_XX.json`.

## Notes

- Prefer lower `overall_mae`.
- If metrics tie, prefer higher `within_tolerance_rate`.
- Do not return null metrics when eval JSON exists in logs.
- `failure_report_path` must be passed to next skills for targeted generation/policy updates.
