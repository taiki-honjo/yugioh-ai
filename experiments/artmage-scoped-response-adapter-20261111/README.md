# Artmage 本体固定＋応答専用の追加部分（2026-11-11）

- 仕様の控え：`spec/artmage-scoped-response-adapter-plan-2026-11-11.md`（`b53aa05f…`）
- 範囲監査の控え：`spec/artmage-response-scope-audit-2026-11-11.json`（`62bad1de…`）
- 報告：`docs/artmage-scoped-response-adapter-2026-11-11-report.md`

| ファイル | 内容 |
| --- | --- |
| `models/frozen/` | 固定したbase・chain-head（元ファイルとバイト一致、`b9c7751f…`・`77191d5d…`） |
| `registration.json`／`amendment-1.json` | 登録（`b5c9128e…`）／キャッシュと事前検査の固定（`b3eb1f29…`） |
| `cache/targets.npz`・`cache/prepare.json` | S1・S2の応答画面の固定logitsなど／学習前の検査結果 |
| `train/candidate/` | 3ファイルの候補：base・chain-head（固定コピー）＋ `response-adapter.flax_model`（`405d2987…`）＋ `manifest.json`（`1617ad7f…`） |
| `train/updates.jsonl`・`train/result.json` | 512更新の記録と、学習後の検査 |
| `eval-runs.jsonl` | 実エンジンの台帳（上限28、使用25） |
| `runs/`・`eval/retain/candidate/` | A・B・Cの記録。`*.policy.json` と `record.json` の `policy` には、読んだ3つのハッシュと、追加部分が働いた画面を記録 |
| `code-changes-sa_runs.diff` | 応答学習の `rl_runs.py` からの変更 |
| `summary.json` | 4者比較（`sa_summarize.py`） |

## 再実行

STOPは `now` のまま置いておく。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。すべてCPUで動く。

エンジン・学習を使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_scoped_response_adapter
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-scoped-response-adapter-20261111 && /opt/ygo-ai/.venv/bin/python -B code/sa_summarize.py
```

実行に使ったコマンド：

- 登録：`code/sa_register.py register`
- 事前検査とキャッシュ：
  - `code/sa_launch.py prepare`
  - `code/sa_register.py fix-cache`
- 学習：`code/sa_launch.py train`
- 評価：
  - `code/sa_launch.py runs a-record|a-replay S1 S2`
  - `code/sa_launch.py runs b-record|b-replay BV1-1cost-VARNISH-OM BV1-2cost-ASSAULT-OM BV1-1cost-ASSAULT-OM BV1-2cost-VARNISH-OM`
  - `code/sa_launch.py runs retain Artmage-g06-p0 … F6-V`

再実行に関する制約：

- 登録・キャッシュ・学習結果・記録済みの条件は、上書きを拒否する。
- 学習は1候補だけ。
- 実エンジンの残り3回は、条件を変えないハーネス修正の後に再実行する場合だけに使える。
