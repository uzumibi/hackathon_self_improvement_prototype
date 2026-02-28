# Skill: generate_targeted_data_from_failures

## Goal

Generate additional training pairs targeted at weak axes using failure examples.

## Inputs

- `focus_axis`
- `failure_report_path`
- `num_samples`

## Steps

1. Read failure examples from `failure_report_path`.
2. Extract linguistic patterns from failed requests.
3. Run generator with env:
   - `NUM_SAMPLES={num_samples}`
   - `APPEND_DATASET=1`
   - `FOCUS_AXIS={focus_axis}`
   - `FAILURE_HINTS_PATH={failure_report_path}`
4. Validate JSONL append success.

## Success criteria

- dataset grows by `num_samples`
- new samples contain clearer cues for `focus_axis`
