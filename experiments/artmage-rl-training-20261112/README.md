# Artmage 実戦学習（PPO＋旧40系列の復習）（2026-11-12）

- 仕様の控え：`spec/artmage-rl-training-plan-2026-11-12.md`（`c044e72a…`）
- 報告：`docs/artmage-rl-training-2026-11-12-report.md`
- 1seed・1候補の初期実験。本番の採用・通常運転の再開はしていない。

| ファイル | 内容 |
| --- | --- |
| `registration.json` | 登録（`06026513…`）。コード・ピン・seed・データ・予算・評価設計・基準側の再利用照合 |
| `amendment-1.json` / `amendment-1.diff` / `amendment-1-old/` | 変更記録1（`e24ad7ec…`、評価前）。学習後の後片付けのassert（ハーネスの不具合）と、ファイルのみの確認による昇格 |
| `data/seeds.json` | 学習seed 256個、評価seed 64個（重複0、`7cf1f54e…`） |
| `models/start/` | 開始モデル＝固定相手（`b9c7751f…` / `77191d5d…`、読み取り専用） |
| `train/` | `episodes.jsonl`・`windows.jsonl`・`minibatches.jsonl`・`progress.json`・`visibility-events.jsonl`・`result.json`（記録のまま、status not_established）・`stopped-state/`・`promotion.json`・`candidate/`（`7f4ee702…` / `4f099640…`） |
| `train-runs.jsonl` | 学習ジョブ1回（GPU） |
| `eval-runs.jsonl` | 新規エンジン対局の台帳（上限141、使用141） |
| `eval/retain/candidate/` | 展開13件（RT1 g06〜g11、F0-V〜F6-V） |
| `eval/battles/` | 128戦：`games.jsonl`（1行1戦）、`games/*.json`（行動記録・ターン開始時の公開盤面・監査）、`policies.json` |
| `summary.json` | 集計（`rlt_summarize.py`） |

## 再実行

STOPは `now` のまま置く。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。

エンジンを使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_rl_training
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-rl-training-20261112 && /opt/ygo-ai/.venv/bin/python -B code/rlt_summarize.py
```

実行に使ったコマンド（順番どおり）：

1. 登録：`code/rlt_register.py register`
2. 学習：`code/rlt_launch.py train`
3. 変更記録1の後、確認と昇格：`code/rlt_promote.py`（CPU。duelは作らない）
4. 展開：`code/rlt_launch.py eval retain Artmage-g06-p0 … F6-V`
5. 対戦：`code/rlt_launch.py eval battles`

学習は1回だけ、評価の台帳は上限に達している。どちらも、もう一度実行することはできない。
