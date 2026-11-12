# Artmage 無限泡影にディアクトロス②を使う損得の機能比較（2026-11-11）

- 仕様の控え：`spec/artmage-imperm-tradeoff-plan-2026-11-11.md`（`a5a530d6…`）
- Codexの監査：`spec/artmage-imperm-target-history-audit-2026-11-11.json`（`a9c7c39f…`）
- 報告：`docs/artmage-imperm-tradeoff-2026-11-11-report.md`
- 手順で操作する機能試験。モデルの判断・学習は使っていない。

| ファイル | 内容 |
| --- | --- |
| `registration.json` | 登録（`02862b88…`）。A-I-Nの再利用照合、対象の乙女の使用履歴の確認を含む |
| `eval-runs.jsonl` | 実エンジンの台帳（上限14、使用10） |
| `runs/<条件>/` | `record.json`・`observations.npz`（保存のみ）・`replay.json`（A-I-P、A-IM-N、A-IM-P、B-N、B-P） |
| `summary.json` | 条件別の集計（`it_summarize.py`）。A-I-Nは、coverageの記録をstep 161で切って参照 |

## 再実行

STOPは `now` のまま置いておく。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。

エンジンを使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_imperm_tradeoff
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-imperm-tradeoff-20261111 && /opt/ygo-ai/.venv/bin/python -B code/it_summarize.py
```

実行に使ったコマンド：

- 登録：`code/it_register.py register`
- 記録：`code/it_launch.py runs record A-I-P A-IM-N A-IM-P B-N B-P`
- 再現：`code/it_launch.py runs replay A-I-P A-IM-N A-IM-P B-N B-P`

再実行に関する制約：

- 記録済みの条件は、もう一度実行できない。
- 残り4回のうち2回は、条件を変えないハーネス修正の後に再実行する場合だけに使える。
