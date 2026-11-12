# Artmage 応答学習＋文脈の保持（2026-11-11）

- 仕様の控え：`spec/artmage-response-context-retention-plan-2026-11-11.md`（`9b4bad58…`）
- 報告：`docs/artmage-response-context-retention-2026-11-11-report.md`

| ファイル | 内容 |
| --- | --- |
| `registration.json` | 学習・実行前の登録（`0d7534c8…`） |
| `amendment-1.json` | 参照分布の固定（学習前、`28916742…`）。コードと条件は変えていない |
| `reference/start-gpu.npz`（`.json`） | 開始モデル自身の、合法手だけで正規化した分布（GPU、`348251bf…`） |
| `reference/cpu-check.json` | 同じ計算をCPUで行った結果との照合 |
| `code-changes-rl_runs.diff` | 応答学習の `rl_runs.py` からの変更（出力先1行だけ） |
| `train/` | `updates.jsonl`（応答CEと保持KLを別々に記録）、`diagnostics.jsonl`、`result.json`、`candidate/`（base `211ad0ee…`、head `8a055055…`） |
| `eval-runs.jsonl` | 実エンジンの台帳（上限28、使用25） |
| `runs/`・`eval/retain/candidate/` | 評価A・B・Cの記録 |
| `summary.json` | 開始モデル／保持なし候補／今回の候補の比較（`cr_summarize.py`） |

## 再実行

STOPは `now` のまま置いておく。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。

エンジン・学習を使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_response_context_retention
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-response-context-retention-20261111 && /opt/ygo-ai/.venv/bin/python -B code/cr_summarize.py
```

実行に使ったコマンド：

- 登録：`code/cr_register.py register`
- 参照分布：
  - `code/cr_launch.py reference`
  - `code/cr_launch.py refcheck`
  - `code/cr_register.py fix-reference`
- 学習：`code/cr_launch.py train`
- 評価：
  - `code/cr_launch.py runs a-record|a-replay S1 S2`
  - `code/cr_launch.py runs b-record|b-replay BV1-1cost-VARNISH-OM BV1-2cost-ASSAULT-OM BV1-1cost-ASSAULT-OM BV1-2cost-VARNISH-OM`
  - `code/cr_launch.py runs retain Artmage-g06-p0 … F6-V`

再実行に関する制約：

- 登録・参照分布・学習結果・記録済みの条件は、上書きを拒否する。
- 学習は1候補だけ。
- 実エンジンの残り3回は、条件を変えないハーネス修正の後に再実行する場合だけに使える。
