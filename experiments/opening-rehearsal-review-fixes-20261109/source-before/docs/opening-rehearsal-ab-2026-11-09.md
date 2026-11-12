# 展開復習の補助損失：有限A/B比較の記録

2026年11月9日（日本時間）。引き継ぎ資料 `docs/yugioh_ai_claude_handoff_2026-11-09.md` の変更Aの実施記録。

**通常運転は停止したままです。** 本番のSTOP・state・採用モデル・実験記録は変更していません（前後の指紋が一致）。常駐・自動起動は再開していません。学習ループからLLM/APIを呼ぶ仕組みは追加していません。どの候補も採用していません。

## 1. 結論

| 腕（各256判断×3 seed） | 66初手の成功数（基準65） | 保持条件の合格 | 3 seed合計の喪失 | 対戦（候補対採用モデル、確認済み） |
| --- | --- | --- | --- | --- |
| A 現行PPO | 57 / 45 / 60 | 0/3 | 33件 | 25勝39敗（64戦）＋評価不能1回 |
| B 復習 λ=1 | 57 / 56 / 64 | 0/3 | 18件 | 40勝56敗（96戦） |
| B 復習 λ=5 | 60 / 62 / 63 | 0/3 | 10件 | 50勝46敗（96戦） |

- 復習は喪失を**減らした**が、**どの候補も既知の展開を維持できなかった**（全9候補が採用前検査で不合格）。λ=5の喪失件数は、3 seedすべてでAより少なかった（5<8、3<20、2<5）。ただし3 seedの小規模比較で、有意性は主張しない。
- 新しく解けたcase_idは全候補で0件。
- 対戦成績の差は、各候補の95%区間がすべて0.5をまたいでおり、結論できない。
- 診断（4節）：同じ大きさでランダム符号の摂動では**喪失0件**、PPO更新の半分では1〜2件の喪失。**退行の主因は自己対戦PPOの勾配の向き**であり、雑音への脆さや、推論経路と学習経路の不一致ではない（経路の一致はテストで確認済み）。
- 引き継ぎ資料の方針（維持できなければ長時間運転へ進まない）に従い、4,096判断の比較や常駐運転には進んでいない。

## 2. 変更内容

### 2.1 追加・変更したファイル

| ファイル | 内容 | SHA256 |
| --- | --- | --- |
| `scripts/opening_rehearsal.py`（新規） | 復習データの検証・読み込み、許容操作集合CE、unroll、勾配合成、trainer用の差分パッチ | `1eb640b32aa598108bceb8583803cfa9735a6a59c1ccba78f525a43aaba4dfe7` |
| `scripts/opening_rehearsal_ab.py`（新規） | 独立root上の有限A/B実行・診断・集計（本番rootを拒否） | `72b9bad160ca052af089b42a71cf324125669934ad77ff059cdb6175daa05f61` |
| `tests/test_opening_rehearsal.py`（新規） | 10項目 | `546cac7466aae14100e5bbac60ee9b987f36a9a8378b0849b41728716000cbb6` |
| `scripts/continuous_opening_backend.py`（変更） | `initialize()` に、設定で有効なときだけ復習を新snapshotへ固定する分岐（下記） | 変更前 `3789adca…` → 変更後 `d92d65b408dc8d6bee7b775c7642fb607de6a68b5c1f77349cb8415cd5686daa` |
| `docs/interruption-landing-evaluation-design-2026-11-09.md`（新規） | 変更Bの設計（未実装） | — |

`scripts/continuous_learning.py`・`continuous_opening_policy.py`・`evaluate.py`・既存テスト・`config/continuous-elfnotes.json` は変更していない（変更前の控えとハッシュは `experiments/opening-rehearsal-ab-20261109/source-before/`）。

```diff
@@ continuous_opening_backend.py initialize()
     text = patched_trainer((Path(cfg['runtime']) / 'agent/scripts/cleanba.py').read_text())
+    rehearsal = cfg.get('rehearsal') or {}
+    if rehearsal.get('enabled'):
+        # Optional train-split rehearsal loss; pinned into this NEW snapshot only.
+        import opening_rehearsal
+        text = opening_rehearsal.prepare_snapshot(cfg, snapshot, text, C.atomic_copy)
     (snapshot / 'train.py').write_text(text)
@@
+    if rehearsal.get('enabled'):
+        state['opening_rehearsal'] = read(snapshot / 'rehearsal/settings.json')['settings']
```

本番設定には `rehearsal` がないため、本番の動作は変わらない。既存の本番snapshotは再生成されない。

### 2.2 損失

```text
g = ∇L_PPO(新しい自己対戦rollout)  +  λ_bc · ∇CE_acceptable(train.npzの系列)
CE_acceptable(s) = -log Σ_{a ∈ 許容操作(s) ∩ 合法操作(s)} p(a | 自分の履歴)
```

- 各PPOミニバッチ（1更新あたり4つ）ごとに、train系列を2本ずつ一様に復元抽出し、勾配を合算してから既存の `clip_by_global_norm(1.0)` → Adam に渡す。PPO部分（rollout・優位性・mask・補助担当操作の除外）は一切変えていない。
- 損失はbase＋chain-headを合成した最終logitsにかける。base・headの両方に勾配が届くことをテストで確認。
- 各系列は先頭からゼロRNN状態でunroll（1系列＝1対戦の自分の判断列。評価時のリセットと同じ）。損失を付けないステップもRNNの履歴として通す。
- 除外：padding（`weights`）、msg 5/11（配置・表示形式の補助が担当）。合法でない許容操作は集合から外す（今回は0件）。有効ステップに合法な正解が0件ならエラーで停止。
- KL項は未実装（`lambda_kl` が0以外なら拒否）。CLEARの完全実装ではない（V-traceによる過去経験のRL更新、価値の保持、方策KLはいずれもない）。
- 召喚権の使用は損失・採点のどちらにも入れていない。

### 2.3 データの保護

- 読み込めるのは、manifestと同じrunの `dataset/ElfnotesKewlTune/train.npz` だけ。`split` が `train` 以外なら拒否し、manifest記録・設定のハッシュと照合する。train-witnessesの順序・長さ・case_idも照合する。初手の手札（多重集合）がvalidation・holdoutと重ならないことも確認する（重複0、train 136手・validation 51手・holdout 68手）。
- 新しいsnapshotには `rehearsal/train.npz` と `train-witnesses.json` だけを固定する（validation/holdoutは入れない）。trainerは起動時に、snapshot内のハッシュとモジュールの読み込み元を確認する。
- 復習対象：416系列・25,802判断。損失対象は19,178判断（うち許容操作が複数の判断が128）。除外は補助担当6,624判断、padding 24,118。

## 3. 実行条件

| 項目 | 値 |
| --- | --- |
| 実験root | `experiments/opening-rehearsal-ab-20261109/`（腕ごとに `a-ppo/`・`b-rehearsal-l1/`・`b-rehearsal-l5/`） |
| 開始モデル | 採用中の組（base `966cc8b1…`・head `1aaf569d…`）を各腕の `models/initial` に複製し、ハッシュ一致を確認 |
| 学習 | 各256環境判断（4環境×32ステップ×2更新）、ミニバッチ勾配8回、lr 3e-5、その他は本番 `train_cycle()` の引数のまま |
| 学習seed | 10000555（以前の退行試験と同じ）、10000777、10000999。3つの腕で同じseedを使う |
| 復習 | 1ミニバッチ2系列、1回の実行で16系列（損失対象699〜729判断）。抽出seedは20261109と学習seedから決まり、λ=1とλ=5で同じ系列を引く |
| λ候補 | {1, 5}。B λ=1の最初の学習ログ（復習勾配ノルム約2〜3、PPO勾配ノルム約13〜15）だけを見て決め、その実行の展開・対戦結果を読む前に `plan.json` へ記録した |
| 展開検査 | validationの自由実行66初手（評価v1、変更なし）。基準モデルが解いたcase_idをすべて維持することが合格条件（`regression_ok`） |
| 教師強制CE | train・validationで診断のみ（学習・選別には使わない）。holdoutは不使用 |
| 対戦 | 各候補対採用モデル、32戦（同じseedで先後を入れ替える16組）、seed 90000555、eval_envs 4、max_eval_steps 4000 |
| 資源 | 各ジョブは `systemd-run` で MemoryMax 5G・CPU 300%（既存の `run_job()`） |

学習前の基準は3つの腕すべてで65/66（失敗は `cue-reco-jj-zaren-g08-p0`、本番の `opening-baseline.json` と同じcase_id）、Luaエラー0で再現した。

Aの `snapshot/train.py` は本番の `train.py` とバイト単位で一致した（`70cd047e…`）。Bは同じ文書に復習の差分だけを当てたもの（`49b556f0…`）。policy-code・evaluator・manifest・runtimeは本番snapshotと同一だった（`verify_opponent_article_basics.py` はroot固有のパスを書き込むため、パス部分だけが異なる）。

## 4. 結果

### 4.1 ケース別の展開（どこかで結果が変わったcase_idのみ。他の37件は全候補で成功）

| case_id | 基準 | A-555 | A-777 | A-999 | B1-555 | B1-777 | B1-999 | B5-555 | B5-777 | B5-999 |
|---|---|---|---|---|---|---|---|---|---|---|
| rukina-pass-g08-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-pass-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-pass-g10-p0 | ○ | ○ | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ |
| rukina-ash_spirit-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-ash_spirit-g10-p0 | ○ | ○ | × | ○ | ○ | × | ○ | ○ | ○ | ○ |
| rukina-ash_parallelism-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-ash_parallelism-g10-p0 | ○ | ○ | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ |
| rukina-veiler_strelitzia-g08-p0 | ○ | ○ | × | ○ | ○ | × | ○ | ○ | ○ | ○ |
| rukina-veiler_strelitzia-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-veiler_strelitzia-g10-p0 | ○ | ○ | × | ○ | ○ | × | ○ | ○ | ○ | ○ |
| elf-rukina-regina-g08-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| elf-rukina-regina-g09-p0 | ○ | ○ | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ |
| elf-rukina-regina-g10-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| elf-par-fire-cost-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| elf-par-fire-cost-g10-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| tune-reco-g08-p0 | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ |
| tune-reco-g09-p0 | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ |
| tune-stone-sweeper-g09-p0 | ○ | × | × | ○ | × | ○ | ○ | × | ○ | ○ |
| tune-juke-veiler-g08-p0 | ○ | × | × | ○ | × | ○ | ○ | × | ○ | ○ |
| tune-juke-veiler-g09-p0 | ○ | × | × | ○ | × | ○ | ○ | × | ○ | ○ |
| mixed-rukina-cue-g08-p0 | ○ | × | × | ○ | × | × | ○ | × | × | ○ |
| mixed-rukina-cue-g09-p0 | ○ | × | × | ○ | × | × | ○ | ○ | × | ○ |
| mixed-rukina-cue-g10-p0 | ○ | × | × | ○ | × | × | ○ | ○ | × | ○ |
| mixed-cue-spirit-g09-p0 | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ |
| cue-reco-jj-zaren-g08-p0 | × | × | × | × | × | × | × | × | × | × |
| cue-reco-jj-zaren-g09-p0 | ○ | × | ○ | × | × | ○ | × | ○ | ○ | × |
| cue-reco-jj-zaren-g10-p0 | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ |
| cue-mix-jj-track-g08-p0 | ○ | × | × | ○ | × | ○ | ○ | × | ○ | ○ |
| cue-rotary-jj-zaren-g10-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | × |

全66件×9候補の表：`experiments/opening-rehearsal-ab-20261109/per-case-openings.csv`。家族別の集計は各 `run-result.json` の `families`。

- 最も崩れやすいのは `mixed-rukina-cue`（9回中、A 6回・B1 6回・B5 4回失敗）と `tune-juke-veiler`／`tune-stone-sweeper`。
- A・seed 10000555 は以前の退行試験（`continuous-opening-integration-check-20261109-v2`）と同じ8件を失った。重みはビット単位では一致しない（GPU・スレッドの非決定性）が、失う箇所は同じだった。B λ=1・同seedも同じ8件を失った。
- 失敗理由の再実行（`opening-failures.json`）では、全候補の判定が再現した。失敗はすべて例外なしの盤面未達（JJ系の一部は解放・素材条件の不成立）で、補助コードの例外による展開失敗は0件だった。

### 4.2 教師強制での一致（診断のみ）

学習前（全腕共通）：train CE 0.0593／一致率 0.9959、validation CE 0.0847／0.9906。

| 腕・seed | train CE / 一致率 | validation CE / 一致率 |
| --- | --- | --- |
| A-555 / 777 / 999 | 0.0627 / 0.0653 / 0.0688（一致率 0.9935〜0.9937） | 0.0689 / 0.0734 / 0.0912 |
| B1-555 / 777 / 999 | 0.0598 / 0.0575 / 0.0573（0.9943〜0.9956） | 0.0659 / 0.0668 / 0.0885 |
| B5-555 / 777 / 999 | 0.0579 / 0.0566 / 0.0590（0.9948〜0.9952） | 0.0665 / 0.0719 / 0.0867 |

教師強制の指標はほとんど動かない一方で、自由実行では多数のケースを失う。1系列約60〜100判断のうち、分岐を決める少数の判断が反転するだけで展開全体が崩れる。教師強制CEだけでは保持を判定できない。

### 4.3 対戦（候補対採用モデル、32戦、seed 90000555）

| 候補 | 勝-敗-分 | 勝率 | 95%区間 | 自然決着のみ | 異常終了 | 打ち切り |
| --- | --- | --- | --- | --- | --- | --- |
| A-555 | 11-21-0 | 0.344 | [0.110, 0.646] | 0.344 | 0 | 0 |
| A-777 | **評価不能** | — | — | — | — | — |
| A-999 | 14-18-0 | 0.438 | [0.198, 0.701] | 0.438 | 0 | 0 |
| B1-555 | 9-23-0 | 0.281 | [0.073, 0.587] | 0.281 | 0 | 0 |
| B1-777 | 13-19-0 | 0.406 | [0.152, 0.701] | 0.406 | 0 | 0 |
| B1-999 | 18-14-0 | 0.563 | [0.299, 0.803] | 0.563 | 0 | 0 |
| B5-555 | 16-16-0 | 0.500 | [0.247, 0.754] | 0.500 | 0 | 0 |
| B5-777 | 17-15-0 | 0.531 | [0.247, 0.803] | 0.531 | 0 | 0 |
| B5-999 | 17-15-0 | 0.531 | [0.247, 0.803] | 0.531 | 0 | 0 |

全候補で確認済み32/32、タイムアウト疑いによる除外は0件。**A-777は、採用中の配置補助 `center_placement.py` の `PlacementMemory.observe()` が例外で止まり、対戦評価を完了できなかった**（`ValueError: list.remove(x): x not in list`。Remix発動後に記録した2枚以外のカードを選んだときに発生）。この結果は除外ではなく「評価不能」として記録し、再試行や補助の修正はしていない。候補の打ち方が変わったことで、元からあった補助の不具合が表に出たもの。

### 4.4 勾配・時間・メモリ

| 腕 | PPO勾配ノルム（更新平均） | 復習勾配ノルム（λ倍前） | 学習の実時間 | 最大RSS | GPU最大使用量 |
| --- | --- | --- | --- | --- | --- |
| A | —（未記録） | — | 40〜42秒 | 1,918〜1,968 MiB | 2,186〜2,425 MiB |
| B1 | 14.3 / 23.3 / 38.3 | 2.58 / 6.98 / 2.98 | 46〜48秒（初回は152秒） | 2,313〜2,678 MiB | 2,409〜2,510 MiB |
| B5 | 13.9 / 21.4 / 39.2 | 2.62 / 6.21 / 2.90 | 48秒（初回は152秒） | 2,312〜2,707 MiB | 2,458〜2,507 MiB |

- GPU使用量は機械全体の値で、開始前から約1,410 MiBが使用中（他プロセス分）。
- 初回の152秒にはJAXのコンパイル約120秒が含まれる。コンパイル後の復習の追加計算は1更新あたり約0.8秒。
- headへの復習勾配は0.003〜0.009と小さい（headの出力はチェーン中だけ効くため）。
- 合算後の勾配は全体ノルム1.0でクリップされ、λ=1では復習成分が全体の約15%にとどまる。

展開検査は1回約42秒（RSS 約990 MiB）、教師強制診断は約30秒、対戦32戦は約70秒。

### 4.5 原因の切り分け（学習なしの診断）

`experiments/opening-rehearsal-ab-20261109/diagnostics/`。Aの候補の更新量Δを使い、評価だけを行った。

| 元にした候補 | θ₀＋0.5Δ（同じ向きで半分） | θ₀＋\|Δ\|×ランダム符号（同じ大きさ） |
| --- | --- | --- |
| A-555（57/66） | 63/66（喪失2件） | **65/66（喪失0件）** |
| A-777（45/66） | 64/66（喪失1件） | **65/66（喪失0件）** |

この大きさの変化そのものに脆いわけではない。PPOの勾配の向きへ進むほど展開が失われる。自己対戦の報酬が求める方向と、先攻展開の手順が衝突している可能性が高い。推論経路と学習経路の一致は、テストで別途確認した（復習unrollが、評価時の `Policy.infer` の逐次推論とも、PPOの多ステップ経路とも数値一致）。

## 5. ハッシュ

| 対象 | SHA256 |
| --- | --- |
| 開始base／head（採用モデル、変更なし） | `966cc8b1c3f0c092e824cc52dffcaf9c71f0c9373ece0fd07831d4a9df4f7cd1` ／ `1aaf569daf53a4453ef2f2e450c924af014aea83b3e4f98fea2644b795b52dc8` |
| train.npz（復習に使用） | `0ef4700c92ddb6f26c3e3f5dc04d4893d2cb02c096304af0bbdc546f8070e4da` |
| validation.npz（教師強制の診断のみ） | `d2c400afea0f0b512efa8f0ed8b97d8f71988345f193fc247024120fa21e6ced` |
| holdout.npz（不使用・変更なし） | `16140b34861f6bc96c4c18f78cf9e66f78ae20c0a9e0e38ff7b9b35a487122de` |
| opening manifest | `7e50140dee233583cb89f0ad3e0197ebc2668901e545532ce6a55bb9e0e69e01` |
| trainer A／B | `70cd047ef8acb7b22f1a82191c1dd03d99770ee7185f88f018c4a2ebfd9399bd` ／ `49b556f08a4783a084dbe95f430def480f0b3acba8c0eb672abb9a06a80e9ea2` |
| native／語彙／デッキ（各ジョブで `runtime_check` 済み） | `19a60f38…` ／ `aeb5be89…` ／ `6e6344d9…`（`env/continuous-opening-runtime.json` の値） |

候補モデル（いずれも不採用）：

| 腕 | seed | base | head |
| --- | --- | --- | --- |
| A | 10000555 | `e4951ae5552687101163c3614ac375c0e00623ecbedb07c90458dd2127f96897` | `e7c84660b6f6bbe4b3f7c5c71e7d3fedee23a9d21ef507d81a96a4533abf35d6` |
| A | 10000777 | `04cb774835e7529e8a6862d056fc3927317b3332a7d053d728beea18cd869142` | `61c73f6ad1109b2afd354b9404c442816e5579e7b43c29fedc4658698324720a` |
| A | 10000999 | `15dfc98bdb7135b31d31ab0ac2bc1a13c852d85cd5ebccb34eb39bdcaf2ca6ce` | `f2bf4fcbbb005cfcfa94856ce11dd72c095456ea2e131385e76d0248d0b18361` |
| B1 | 10000555 | `195c025ccc0c830ecac04d7f195188d335f847c4d458b65c4c517481de24c152` | `caba63f2cf5f1c1ab58c5ab61fefa3769b2aa16af11e7f1d3e21c46aebccf655` |
| B1 | 10000777 | `1bf45491e023a4b7f7dffb322e40ad60553910c16d1fd2e928f9d2e834481faf` | `02ff92a063fd23cd6fced23ca391056fa2f6204f56399ed929f3d4e2d7d5948b` |
| B1 | 10000999 | `836107ea964bd4c37a28cb2620eee9b6aea1bcbcf263b9756208339d8c089fe2` | `036d565eb398cc143bace617ea875a54d9b6c4368ab611b2437a7b4fd53cdcdf` |
| B5 | 10000555 | `c5f60622670c428ad901bb266b4aec677d5593ad40610b263f8ea2f1bbddd1d0` | `8eb1e4e9d3a81e00662f0cbc1150d6419a568ce2c90d068ba39a69bc7cc95f49` |
| B5 | 10000777 | `e8c677e1cf457888c7cf0aa908e8fd911b357c432fd7cd0d2783820685fd8a7f` | `7868c60a2820881c1baa519570fc57f752d4b04d380c97d7b4c0f987b12b04f4` |
| B5 | 10000999 | `e751962884e132fbe4e68685b87279984f76080151d198bbeb032ff2c81a2894` | `1a791a82e9e08f6a3b6b5ab51a5db5b78c5bb61164d8d7e08312977252fbb95a` |

## 6. 検証

```bash
cd /home/hatk8/projects/yugioh-ai
/opt/ygo-ai/.venv/bin/python -B -m unittest discover -s tests -p 'test_continuous*.py' -v   # 既存11項目：合格
/opt/ygo-ai/.venv/bin/python -B -m unittest tests.test_opening_rehearsal -v                  # 新規10項目：合格（約75秒、CPU）
```

新規テストで確認した内容：

- 複数の許容操作を集合として扱う。padding・補助担当の操作・違法な正解は損失に入らず、logitsを変えても損失が変わらない。
- 合法な正解のない有効ステップ、lengthsと合わないweightsは拒否する。
- validation／holdoutの指定、validation.npzを `train.npz` に改名したもの、固定ハッシュの不一致、手札の分割の重なりを拒否する。
- 抽出が決定的であること。
- 実モデル・実データでの経路一致（評価時の逐次推論、PPOの多ステップ経路、系列境界ごとのRNNリセット）。
- base・headへの勾配到達と、`PPO勾配＋λ×復習勾配` の合算。採用モデルのファイルが変わらないこと。
- 無効時のtrainerが本番とバイト一致し、有効時は両方の損失箇所に補助担当maskが残ること。
- 新rootの初期化でtrainだけが固定されること（本番の指紋は不変）。
- 実験ランナーが本番rootとその親を拒否すること。

生成されたsnapshotを実際に使った証拠：各学習ログの `OPENING_REHEARSAL_DATA` と `OPENING_REHEARSAL_UPDATES 2`、`rehearsal-metrics.jsonl`（抽出した系列番号・勾配ノルム）。Aの学習ログには復習の出力がないことも確認した。

## 7. 本番保護の確認

`production-fingerprint-before.json` と `-after.json` が一致した。対象：STOP（内容 `now`）、state（`stopped`、pid null、best＝`models/initial`）、opening-baseline、snapshot/hashes、採用モデル2ファイル、設定2つ。実験の各腕は終了後に `STOP` と `stopped` 状態にした。Windows側のコントローラーやStartupには触れていない。

## 8. 残る課題と次の候補（未実施・要判断）

1. **保持は未達成。** 小さな復習損失を足すだけでは、自己対戦PPOの勾配の向きに勝てない。次に検討する候補（いずれも有限・独立rootで、1つずつ比較する）：
   - λをさらに大きくする（例：20）。PPOの学習成分がほぼ消える可能性が高く、「PPOを弱めただけ」と区別できる対照が要る。
   - PPOの学習率を下げる、または1区間の更新回数を減らす（摂動診断で、同じ向きでも半分なら喪失が1〜2件に減った）。
   - 自由実行で決定的な判断（分岐点）の復習を重くする、展開系列を多めに抽出する。
   - 任意のKL（固定した採用モデルを参照、`lambda_kl`）を別段階で追加する。
   - 先攻展開の部分を凍結し、対戦で学習する部分を分ける構成。
2. **採用中の配置補助の不具合**（`center_placement.py` の `PlacementMemory.observe`、Remix対象の記憶）。学習候補が補助を例外へ追い込むことがある。本番の評価で起きればジョブ失敗になる。補助の修正は別変更として扱い、配置の回帰試験44/44などで確認する必要がある。今回は修正していない。
3. 3 seed・256判断の小規模比較のため、λ=5がAより良いという向きにも統計的な保証はない。対戦の差も結論できない。
4. Aの勾配ノルムはtrainerを変えないため記録していない（Aをバイト一致させることを優先した）。
5. 評価の変更B（妨害後の複数の着地点）は設計のみ：`docs/interruption-landing-evaluation-design-2026-11-09.md`。今回の比較はすべて評価v1で行っており、採点の変更と同時には比較していない。
6. holdoutの88件は使っていない。最終性能の確認には、新しい固定試験を別に用意するのが望ましい。

## 9. 再実行

```bash
cd /home/hatk8/projects/yugioh-ai
/opt/ygo-ai/.venv/bin/python -B -u scripts/opening_rehearsal_ab.py run --plan experiments/opening-rehearsal-ab-20261109/plan.json
/opt/ygo-ai/.venv/bin/python -B -u scripts/opening_rehearsal_ab.py diagnose --plan experiments/opening-rehearsal-ab-20261109/plan.json --arms a-ppo --seeds 10000555,10000777
/opt/ygo-ai/.venv/bin/python -B -u scripts/opening_rehearsal_ab.py summarize --plan experiments/opening-rehearsal-ab-20261109/plan.json
```

完了済みの段階はファイルから再利用する。新しい実行は新しい実験root（新しい `experiment_root`）で行うこと。
