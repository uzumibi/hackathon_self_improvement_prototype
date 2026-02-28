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
5. If W&B MCP is available, run tools with **schema-correct arguments** and scoped projects:
   - First resolve target scope via `query_wandb_entity_projects` and select only:
     - model/runs project: `music-params-mcp-loop`
     - weave traces project: `music-params-eval`
   - Exclude unrelated projects (e.g. `patchtst_*`) from analysis context.
6. Call each tool with correct argument names:
   - `query_wandb_entity_projects`:
     - `{"entity": <selected_entity>}` (or `{}` as fallback)
   - `query_wandb_tool` (GraphQL is required):
     - include `query` + `variables` (entity/project/limit)
     - retrieve recent runs and `summaryMetrics` for before/after comparison
   - `count_weave_traces_tool`:
     - `{"entity_name": ..., "project_name": ..., "filters": {"status": "error", "trace_roots_only": true}}`
   - `query_weave_traces_tool`:
     - `{"entity_name": ..., "project_name": ..., "filters": {"status": "error", "trace_roots_only": true}, "columns": [...], "limit": 10}`
   - `query_wandb_support_bot`:
     - `{"question": "..."}`
   - `create_wandb_report_tool`:
     - `{"entity_name": ..., "project_name": ..., "title": ..., "description": ..., "markdown_report_text": ...}`
7. Record both success and failure details for each tool, including attempted arguments and error text.
8. Store MCP tool outputs as `skill_wandb_mcp_inspection` in `generated_skills/iter_XX.json`.

## Notes

- Prefer lower `overall_mae`.
- If metrics tie, prefer higher `within_tolerance_rate`.
- Do not return null metrics when eval JSON exists in logs.
- `failure_report_path` must be passed to next skills for targeted generation/policy updates.
