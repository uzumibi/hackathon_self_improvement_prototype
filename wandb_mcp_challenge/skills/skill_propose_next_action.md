# Skill: propose_next_action_from_metrics

## Goal

Decide the next training and data-generation action from evaluation metrics.

## Inputs

- `current_state`: current hyperparameters and data knobs
- `best_mae_so_far`
- `eval_summary`

## Policy

1. If `overall_mae` improves by at least `0.01`, keep hyperparameters.
2. Otherwise:
   - increase `training_steps` by +10 (cap 120)
   - reduce `learning_rate` by 20% (floor 1e-5)
3. Always set `focus_axis = worst_axis` if available.
4. Increase `num_samples` by +4 (cap 40) when worst axis is found.

## Output format

```json
{
  "training_steps": 45,
  "learning_rate": 8e-5,
  "num_samples": 12,
  "focus_axis": "nostalgia",
  "reason": "nostalgia axis underperforming"
}
```
