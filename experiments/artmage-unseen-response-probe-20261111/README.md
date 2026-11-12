# Artmage 未学習の手札誘発への応答診断（2026-11-11）

- 仕様の控え：`spec/artmage-unseen-response-probe-plan-2026-11-11.md`（`234db9b7…`）
- 報告：`docs/artmage-unseen-response-probe-2026-11-11-report.md`
- 診断のみ。学習0回、モデル更新0回。

| ファイル | 内容 |
| --- | --- |
| `models/base/` | 元のbase・chain-head（固定コピー） |
| `models/candidate/` | coverage候補の3ファイルとmanifest（バイト一致コピー、adapter `00977941…`） |
| `registration.json`／`amendment-1.json` | 登録（`bc89fde1…`）／機能対照の再利用判定（`365590fa…`。エンジンは使っていない） |
| `eval-runs.jsonl` | 実エンジンの台帳（上限26、使用20） |
| `runs/<条件>/` | `record.json`・`observations.npz`・`replay.json` |
| `summary.json` | 応答画面ごとの元logits、補正、パスとの相対補正、得点差、結果（`up_summarize.py`） |

## 再実行

STOPは `now` のまま置いておく。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。

エンジンを使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_unseen_response_probe
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-unseen-response-probe-20261111 && JAX_PLATFORMS=cpu /opt/ygo-ai/.venv/bin/python -B code/up_summarize.py
```

実行に使ったコマンド：

- 登録：`code/up_register.py register`
- 方策比較：`code/up_launch.py runs record|replay` に `H1|H2-ASH|GAMMA-BASE|CAND` の8条件を渡す
- 機能対照の再利用判定：`code/up_register.py controls`
- 機能対照：`code/up_launch.py runs record|replay H1-ASH-CTRL-G H1-GAMMA-CTRL-G`

再実行に関する制約：

- 記録済みの条件は、もう一度実行できない。
- 実エンジンの残り6回のうち2回は、条件を変えないハーネス修正の後に再実行する場合だけに使える。
