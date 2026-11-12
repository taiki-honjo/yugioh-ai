# 配置補助v2の本番反映

2026年11月9日 14:00（日本時間）。依頼資料：`experiments/placement-v2-adoption-20261109/claude_v2_and_next_experiment_2026-11-09.md`

**反映したのは配置補助の修正版（v2）だけです。** ニューラル重みは採用中の base＋chain-head の組のままで、新しい学習候補は採用していません。通常運転は停止したままで、ログオン時の起動とWindows側のファイルには触れていません。旧版のA/B表と「評価不能」の記録は変更していません。

## 1. 手順と結果

| 手順 | 結果・証拠 |
| --- | --- |
| 旧本番の保持 | 旧root `experiments/continuous-elfnotes-openings-20261109/`（STOP・state・snapshot・モデル）を変更していない。作業前の全ファイルのハッシュは `experiments/placement-v2-adoption-20261109/before.sha256`、旧設定の控えは `source-before/config/` |
| 新しい本番rootの作成 | `experiments/continuous-elfnotes-openings-v2-20261109/`。他のファイルより先に `STOP=now` を置き、state は `stopped` |
| 新snapshotへの固定 | 配置補助v2、同じ攻守表示補助、同じモデル2つ、デッキ・語彙・native・評価器。旧rootのsnapshotとの差は4ファイルだけ：`center_placement.py`（v1→v2）、`opening-policy.json`（v2の方針設定）、`verify_opponent_article_basics.py`（root自身のパスを含む）、`continuous_opening_backend.py`（下記のモジュール出所の検査を追加）。学習コード `train.py` は同一（`70cd047e…`） |
| 新rootでの基準 | 66初手で65/66。旧rootの基準とcase_id・ケース単位の結果が一致、Luaエラー0（`v2-root-check.json`）。STOPを外さないよう、評価ジョブは直接起動した |
| 本番設定の切り替え | `config/continuous-elfnotes.json` の `root` を新rootに、`verified_policy_config` を v2の方針設定に変更（`switch-record.json`） |

| 対象 | 切り替え前 | 切り替え後 |
| --- | --- | --- |
| `config/continuous-elfnotes.json` | `5a6fab46…` | `354ff515af60c4e6d011ac4e45f7d8b7d0b479309e4c5a5e608d40aa1950114a` |
| 方針設定 | `latest-verified-opening-policy.json`（`70cfef5b…`、v1、変更なし） | `verified-opening-policy-placement-remix-fix-20261109.json`（`bda5fb0b059727b0e69b315eeb33757da8bc80efcf7edcc2705f2b0dae2aadde`） |
| `center_placement.py` | v1 `805e6821…` | v2 `3e96241095487fb48426099d774b7813cee348fc282dacbde0ced9af2d7abbba` |
| base／head | `966cc8b1…`／`1aaf569d…` | 同じ |
| 本番root | `continuous-elfnotes-openings-20261109`（停止、保持） | `continuous-elfnotes-openings-v2-20261109`（停止） |

## 2. 推論の入口

| 入口 | 読み込み元 | 確認結果 |
| --- | --- | --- |
| `scripts/continuous_opening_policy.py` | 本番設定 → 本番rootの `best_checkpoint` と、snapshot内の補助コード | v2（`3e962410…`）、採用モデルの組、検証用の初手1件が成功 |
| `scripts/verified_opening_policy.py`（`scripts/deck_policy.py` がエルフェンノーツ用に呼ぶ） | **変更**：これまでは `latest-verified-opening-policy.json` を直接読んでいた。現在は本番設定の `verified_policy_config` を読む | v2、採用モデルの組、検証用の初手1件が成功 |

2つの入口が同じ版を読むよう、`verified_opening_policy.py` を変更した。片方の設定だけを替えても、全入口がv2になるわけではないため。

別版のモジュールが既に読み込まれたプロセスでは、どちらの入口も読み込みを拒否するようにした（「use a fresh process」）。`continuous_opening_backend.create_policy()` と `AdapterBank` は、補助モジュールがそのrootのsnapshot以外から読み込まれていれば停止する。`verified_opening_policy.py` は、設定したフォルダ以外からの読み込みと、`center_placement.py` のハッシュ不一致を拒否する。

これらは `tests/test_policy_entries.py`（4項目、プロセスを毎回新しく起動）で確認した。

`evaluation/all-deck-position-20261109/` 等の過去の評価記録に含まれるコードの写しは変更していない。ただし、これらが `scripts/verified_opening_policy.py` を呼んで再実行された場合は、v2が読み込まれる。

## 3. 切り戻し

`experiments/placement-v2-adoption-20261109/source-before/config/continuous-elfnotes.json` を `config/continuous-elfnotes.json` へ戻せば、両方の入口がv1を読む。旧rootは一度も変更していない。

## 4. 適用範囲と注意

- 176件の一致と、停止していた対戦の完了は、今回の修正を採用する根拠になる。ただし、あらゆる対戦で配置が最適であることの証明ではない。
- 教師の手本の生成（`PlacementTeacher`）には、公開チェーン長を渡していない。チェーン終了で記録を終える規則は、教師には適用されない。
- 旧A/B実験（`opening-rehearsal-ab-20261109`）の記録はv1のまま。新しい本番rootに合わせて本番の指紋の項目を変えたため、旧実験の `diagnose` は「本番が変わった」として実行を拒否する。旧実験は完了済み。
- ログオン時のショートカット（`YugiohAI-Continuous-Elfnotes.lnk`）は `C:\Users\hatk8\AppData\Local\YugiohAI\continuous-elfnotes.ps1 -Action Run` を指している。WSLからはこのスクリプトが見つからなかった。仮に起動されても、新旧どちらのrootにも `STOP=now` があるため、ワーカーは停止状態で終わる。
