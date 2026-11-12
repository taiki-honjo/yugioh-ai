# Artmage 応答専用モデルに維持例を加えた比較（2026-11-11）

- 仕様の控え：`spec/artmage-response-coverage-plan-2026-11-11.md`（`502538df…`）
- 報告：`docs/artmage-response-coverage-2026-11-11-report.md`

| ファイル | 内容 |
| --- | --- |
| `models/frozen/` | 固定したbase・chain-head（元ファイルとバイト一致） |
| `code/sa_model.py` | 推論コード。前回の領域と同一（`fe9856bd…`） |
| `registration.json`／`amendment-1.json` | 登録（`2f0d2d24…`）／キャッシュと事前検査の固定（`9ad1852b…`） |
| `cache/screens.npz`・`cache/prepare.json` | 4つの学習画面（S1・S2・M・P）の固定logitsと参照分布／事前検査 |
| `train/candidate/` | 3ファイルの候補（追加部分 `00977941…`、manifest `48d6e2c7…`） |
| `train/updates.jsonl`・`train/result.json` | CE項とKL項を別々に記録した512更新／学習後の検査 |
| `eval-runs.jsonl` | 実エンジンの台帳（上限18、使用16） |
| `runs/B-*` | 8条件の記録・再現 |
| `code-changes-cv_runs.diff` | 前回の `sa_runs.py` からの変更 |
| `summary.json` | 集計（`cv_summarize.py`） |

- Aは、Bの2条件から参照している（照合結果は `summary.json` の `A_reference`）。
- Cは、新しく実行せずに再利用した（根拠は `registration.json` の `evaluation.C`）。

## 再実行

STOPは `now` のまま置いておく。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。すべてCPUで動く。

エンジン・学習を使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_response_coverage
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-response-coverage-20261111 && /opt/ygo-ai/.venv/bin/python -B code/cv_summarize.py
```

実行に使ったコマンド：

- 登録：`code/cv_register.py register`
- 事前検査とキャッシュ：
  - `code/cv_launch.py prepare`
  - `code/cv_register.py fix-cache`
- 学習：`code/cv_launch.py train`
- 評価：`code/cv_launch.py runs b-record|b-replay` に8条件の名前を渡す

再実行に関する制約：

- 登録・キャッシュ・学習結果・記録済みの条件は、上書きを拒否する。
- 実エンジンの残り2回は、条件を変えないハーネス修正の後に再実行する場合だけに使える。
