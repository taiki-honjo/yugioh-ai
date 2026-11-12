# Artmage 未指定の応答の相対得点差を保つ比較（2026-11-11）

- 仕様の控え：`spec/artmage-relative-score-retention-plan-2026-11-11.md`（`20200c65…`）
- 報告：`docs/artmage-relative-score-retention-2026-11-11-report.md`

| ファイル | 内容 |
| --- | --- |
| `registration.json` | 登録（`b2aca25c…`） |
| `amendment-1.json` | キャッシュと事前検査の固定（`6bde184f…`） |
| `amendment-2.json` | 全行動照合の後に、実行が必要な条件を確定（`729f5fab…`） |
| `cache/screens.npz` | 4画面。coverageのキャッシュと配列が一致し、パスの位置と保持マスクを追加 |
| `train/candidate/` | 3ファイルの候補（adapter `c110f964…`、manifest `3266c90d…`） |
| `check.json` | 既存12記録の全行動照合（エンジンは使っていない） |
| `eval-runs.jsonl` | 実エンジンの台帳（上限30、使用12） |
| `runs/` | 新規の記録・再現：襲撃側4条件、幽鬼うさぎ2方策 |
| `code-changes-*.diff` | 元のコードからの差分（学習・実行の2種類） |
| `summary.json` | 集計（`rs_summarize.py`） |

## 再実行

STOPは `now` のまま置いておく。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。すべてCPUで動く。

エンジン・学習を使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_relative_score_retention
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-relative-score-retention-20261111 && JAX_PLATFORMS=cpu /opt/ygo-ai/.venv/bin/python -B code/rs_summarize.py
```

実行に使ったコマンド：

- 登録：`code/rs_register.py register`
- 事前検査とキャッシュ：
  - `code/rs_launch.py prepare`
  - `code/rs_register.py fix-cache`
- 学習：`code/rs_launch.py train`
- 全行動照合：`code/rs_launch.py check`（追補2を書く）
- 評価：
  - `code/rs_launch.py flc record|replay` に襲撃側4条件の名前を渡す
  - `code/rs_launch.py probe record|replay H1-USAGI-BASE H1-USAGI-CAND`

再実行に関する制約：

- 登録・キャッシュ・学習結果・照合結果・記録済みの条件は、上書きを拒否する。
- 実エンジンの予備2回は、条件を変えないハーネス修正の後に再実行する場合だけに使える。
