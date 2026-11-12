# Artmage 相手ターンの温存と応答（2026-11-11）

- 仕様の控え：`spec/artmage-opponent-turn-probe-plan-2026-11-11.md`（`df8a8b5e…`）
- 報告：`docs/artmage-opponent-turn-probe-2026-11-11-report.md`

| ファイル | 内容 |
| --- | --- |
| `registration.json` | 実行前の登録（`5591eefa…`） |
| `reuse.json` | 旧記録を再利用する根拠。エンジンを使わない照合（`8c799b59…`） |
| `play-vs-comparison.diff` | 旧実行コードとの差分（デュエル処理の本体） |
| `eval-runs.jsonl` | 専用台帳（上限12） |
| `runs/<条件>/` | `record.json`・`observations.npz`・`replay.json` |
| `summary.json` | 条件別の集計（`otp_summarize.py`） |

## 再実行

STOPは `now` のまま置いておく。起動スクリプトが実行中だけ外し、終了・失敗時に戻す。

エンジンを使わない確認：

```bash
cd /home/hatk8/projects/yugioh-ai && /opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_artmage_opponent_turn_probe
```

```bash
cd /home/hatk8/projects/yugioh-ai/experiments/artmage-opponent-turn-probe-20261111 && JAX_PLATFORMS=cpu /opt/ygo-ai/.venv/bin/python -B code/otp_summarize.py
```

- `reuse.json` と `registration.json` は、上書きを拒否する（作り直す場合は別の出力先を用意する）。

エンジンを使うもの（台帳の上限を消費する。現在10/12）：

```bash
/opt/ygo-ai/.venv/bin/python -B code/otp_launch.py runs record S1-D1R0 S1-D0R1 S1-D1R1 S2-D1R0 S2-D1R1
```

```bash
/opt/ygo-ai/.venv/bin/python -B code/otp_launch.py runs replay S1-D1R0 S1-D0R1 S1-D1R1 S2-D1R0 S2-D1R1
```

- 記録済みの条件は、もう一度記録できない（実行時に拒否される）。
- 残りの2回は、ハーネスの不具合を直した後にやり直す場合だけに使える。
