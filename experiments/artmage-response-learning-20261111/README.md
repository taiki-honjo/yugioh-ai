# Artmage 応答1手の小規模学習（2026-11-11）

- 仕様の控え：`spec/artmage-response-learning-plan-2026-11-11.md`（`51fc9719…`）
- 報告：`docs/artmage-response-learning-2026-11-11-report.md`

| ファイル | 内容 |
| --- | --- |
| `registration.json` | 学習・実行前の登録（`02871f82…`） |
| `data/manifest.json` | 新しい2系列のmanifest（`2281d191…`）。旧40系列は手札分岐のmanifestを参照 |
| `code-changes-batch_model.diff` | 固定コピーからの変更（更新数と診断時点だけ） |
| `diag/pre-training.json` | 学習前の診断（CPU）：衝突、教師強制、応答の確率 |
| `train/` | `updates.jsonl`・`diagnostics.jsonl`・`result.json`・`candidate/`（base `7d44a779…`、head `a8a176da…`） |
| `train-runs.jsonl` | 学習の台帳（1回） |
| `eval-runs.jsonl` | 実エンジンの台帳（上限28、使用25） |
| `runs/A-*`・`runs/B-*` | `record.json`・`observations.npz`・`replay.json` |
| `eval/retain/candidate/` | 保持13件の記録 |
| `summary.json` | 集計（`rl_summarize.py`） |

## 再実行

STOPは `now` のまま置いておく。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。

エンジン・モデルを使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_response_learning
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111 && /opt/ygo-ai/.venv/bin/python -B code/rl_summarize.py
```

実行に使ったコマンド：

- 登録と診断・学習：
  - `code/rl_register.py`
  - `code/rl_launch.py diag`
  - `code/rl_launch.py train`
- 評価：
  - `code/rl_launch.py runs a-record S1 S2`
  - `code/rl_launch.py runs a-replay S1 S2`
  - `code/rl_launch.py runs b-record|b-replay BV1-1cost-VARNISH-OM BV1-2cost-ASSAULT-OM BV1-1cost-ASSAULT-OM BV1-2cost-VARNISH-OM`
  - `code/rl_launch.py runs retain Artmage-g06-p0 … F6-V`

再実行に関する制約：

- 登録・manifest・学習結果・記録済みの条件は、上書きを拒否する。
- 学習は1候補だけで、もう一度は実行できない。
- 実エンジンの残り3回は、条件を変えないハーネス修正の後に再実行する場合だけに使える。
