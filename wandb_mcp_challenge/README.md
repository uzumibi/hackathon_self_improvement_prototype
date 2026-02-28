# W&B MCP + Skills Self-Improvement Workflow (Cline版)

この構成は、Mini Challenge 要件（eval自動化 / 最適化ループ / generated skills提出）を、**Clineの実行権限を前提にシンプル化**して満たすテンプレートです。

## 1) セットアップ

必要な環境変数:

- `MISTRAL_API_KEY`
- `WANDB_API_KEY`
- `WANDB_ENTITY`（任意だが推奨）
- `WANDB_PROJECT`（未指定時は `music-params-mcp-loop`）

VS Code で W&B MCP サーバーを使う設定は [../.vscode/mcp.json](../.vscode/mcp.json) を参照。

## 2) 構成

- 実行ループ: [mcp_skill_orchestrator.py](mcp_skill_orchestrator.py)
- W&B MCP クライアント: [wandb_mcp_client.py](wandb_mcp_client.py)
- Skills 定義:
  - [skills/skill_eval_and_diagnose.md](skills/skill_eval_and_diagnose.md)
  - [skills/skill_propose_next_action.md](skills/skill_propose_next_action.md)
  - [skills/skill_generate_targeted_data.md](skills/skill_generate_targeted_data.md)
- 生成物:
  - `generated_skills/iter_XX.json`
  - `generated_skills/policy_iter_XX.json`
  - `runs/challenge_report_*.json`
  - `runs/challenge_submission_*.json`
  - `runs/challenge_submission_latest.json`

## 3) 実行

```bash
python3 wandb_mcp_challenge/mcp_skill_orchestrator.py
```

主な制御用環境変数:

- `MCP_LOOP_MAX_ITERS`（既定: `3`）
- `MCP_LOOP_TARGET_MAE`（既定: `0.20`）
- `MCP_LOOP_EVAL_SAMPLES`（既定: `100`）
- `MCP_LOOP_INIT_NUM_SAMPLES` / `MCP_LOOP_INIT_STEPS` / `MCP_LOOP_INIT_LR`

## 4) 自動化される改善ループ

1. `data_generation.py` でデータ生成（失敗例ヒント + policy反映）
2. `ft_music_params.py` で Fine-tuning Job 作成
3. Job完了待ち後、`eval_music_params.py` で評価
4. `overall_mae` と `mae_per_axis` から弱点軸を特定
5. `training_steps / learning_rate / focus_axis / num_samples` を自動更新
6. `agent_policy.json` を弱点軸ベースで自動更新
7. W&B MCP を使って run/metric 情報を取得（可能なツール名を自動探索）
8. 各反復で `generated_skills/iter_XX.json` を保存
9. 最終的に提出用サマリ `challenge_submission_latest.json` を出力

## 5) Mini Challenge 要件との対応

### Proven Improvement ✅

- 各反復の履歴: `runs/challenge_report_*.json`
- 提出用要約: `runs/challenge_submission_latest.json`
- `baseline_mae`, `best_mae`, `absolute_gain`, `relative_gain_percent` を自動集計
- W&B run上でも `overall_mae`, `best_mae`, `mae/*`, `improvement_vs_baseline` を確認可能

### Generated Skills Submitted ✅

- 生成スキル成果物:
  - `generated_skills/iter_*.json`
  - `generated_skills/policy_iter_*.json`
- スキル定義:
  - `skills/*.md`

### Creativity / Completeness

- 失敗例を次イテレーションのデータ生成プロンプトに注入
- 弱点軸を使った policy自己更新
- eval → analysis → improvement を単一オーケストレータで自動接続
- W&B MCPのツール差異に耐えるフォールバック実装（`list_tools`→候補ツール試行）

## 6) Cline前提で簡素化した点

- Copilot向けの間接的なコード自己改変処理を外し、**実験ループ本体に集中**
- 改善ルールを `skills/*.md` と `mcp_skill_orchestrator.py` に明示化
- 提出に必要な証跡（改善値・生成スキル一覧）をサマリJSONとして自動出力

## 7) 提出時に添付すべきファイル

- `wandb_mcp_challenge/runs/challenge_submission_latest.json`
- `wandb_mcp_challenge/runs/challenge_report_*.json`
- `wandb_mcp_challenge/generated_skills/iter_*.json`
- `wandb_mcp_challenge/generated_skills/policy_iter_*.json`
- `wandb_mcp_challenge/skills/*.md`
