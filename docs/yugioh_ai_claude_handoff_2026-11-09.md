# 遊戯王AI：研究メモとClaudeへのコード変更引き継ぎ

作成日：2026-11-09（日本時間）。この日の実ファイルを読み直して作成。

この資料は、ユーザーが今後Claudeに変更を依頼するためのコンテクストです。現状の事実、論文の内容、これからの実装提案を分けています。**提案は未実装です。資料作成に伴うコード変更・学習再開・モデル採用は行っていません。**

## 1. ユーザーの目的と現在の依頼範囲

- 自分のPCで、エルフェンノーツ＋キラーチューンのAIを学習・改善したい。
- 通常の学習運転はPythonなどのコードで完結させ、CodexやClaudeなどのLLM APIを学習ループから呼びたくない。
- 別チャットで学習した先攻展開を引き継ぎつつ、自己対戦で強くしたい。
- 誘発を受けて当初の目標盤面へ届かない場合にも、別ルートや妥協盤面を学べるようにしたい。
- **ユーザーは通常運転の停止を指示しており、その指示は継続中。** コード変更や有限の試験を後日依頼されても、それだけで常駐運転の再開を許可されたとは扱わない。

この資料の推奨方針は「実戦＋復習」です。まず復習による保持を小規模に比較し、次に妨害別の評価を広げます。目標盤面への到達率と実戦の勝率は、別々の指標として記録します。

## 2. 作業環境・アクセス

| 項目 | 値 |
| --- | --- |
| WSLディストリビューション | Ubuntu-22.04 |
| WSLユーザー | hatk8 |
| プロジェクト | `/home/hatk8/projects/yugioh-ai` |
| Windowsからのプロジェクト参照 | `\\wsl.localhost\Ubuntu-22.04\home\hatk8\projects\yugioh-ai` |
| Python | `/opt/ygo-ai/.venv/bin/python` |
| 対象デッキ | `/home/hatk8/projects/yugioh-ai/assets/deck/ElfnotesKewlTune.ydk` |
| この資料のWindows作成元 | `C:\Users\hatk8\Documents\Codex\2026-10-27\new-chat\outputs\yugioh_ai_claude_handoff_2026-11-09.md` |

Python実行ファイルの最終シンボリックリンクを解決しないこと。`Path.resolve()`等でシステムPythonへ変えると、仮想環境の依存関係が失われます。既存の`load_config()`にはその対策があります。

確認時、プロジェクトとその親・docs内に適用する`AGENTS.md`は見つかりませんでした。プロジェクト直下はGitリポジトリではありませんでした。変更前のファイル・設定・ハッシュを別途保存し、変更差分を提示してください。後日の環境変更があれば、実ファイルを優先して再確認します。

## 3. 現在の状態：通常運転は停止中

常時学習用の現在の領域：

```text
/home/hatk8/projects/yugioh-ai/experiments/continuous-elfnotes-openings-20261109/
```

資料作成時の`state.json`は次の状態でした。

```json
{
  "status": "stopped",
  "phase": "idle",
  "completed_cycles": 0,
  "added_steps": 0,
  "pending": null,
  "pid": null,
  "best_checkpoint": "/home/hatk8/projects/yugioh-ai/experiments/continuous-elfnotes-openings-20261109/models/initial",
  "latest_checkpoint": "/home/hatk8/projects/yugioh-ai/experiments/continuous-elfnotes-openings-20261109/models/initial",
  "integration_test_candidate_adopted": false,
  "last_decision": "ready_but_stopped"
}
```

同領域の`STOP`は存在し、内容は`now`でした。確認時、対象の常時学習・snapshot/train.pyの実行プロセスはありませんでした。採用モデルの参照先は初期モデルのままです。

旧領域`/home/hatk8/projects/yugioh-ai/experiments/continuous-elfnotes/`も保存・停止しています。以前の69,632判断モデルと、そこから4,096判断だけ追加学習した重みは、現在の初期モデルへ混ぜていません。

Windows側コントローラー：

```text
C:\Users\hatk8\AppData\Local\YugiohAI\continuous-elfnotes.ps1
```

ログオン時の入口はStartupショートカットです。以前試したWindowsタスクスケジューラーのタスクは無効です。現在の`STOP`を削除したり、`Start`・`resume`・常駐`run`を実行したりしないでください。

現行設定の抜粋です。これらは停止中の通常運転用の値で、初回の比較試験をこの長さで回す指示ではありません。

| 設定 | 現在値 |
| --- | --- |
| backend | `verified_opening` |
| bootstrap_steps / steps_per_cycle | 4,096 / 65,536 |
| learning_rate | 0.00003 |
| eval_envs | 4 |
| eval / reference / confirmation episodes | 128 / 64 / 256 |
| max_eval_steps | 4,000 |
| job_memory_gib / cpu_percent | 5 / 300 |
| job_timeout_seconds / rest_seconds | 1,800 / 60 |
| minimum_free_gib | 8 |
| opening_regression_required | true |

`initial_training_steps=69632`はRL起源の判断数です。別途行った教師あり展開学習を含む総更新数として扱わないでください。

## 4. 採用中のモデルと接続済みの仕組み

参照設定：`/home/hatk8/projects/yugioh-ai/config/latest-verified-opening-policy.json`

学習成果の原本：

```text
/home/hatk8/projects/yugioh-ai/experiments/all-openings-bar-alternative-20261108/
run-20261108-194920/training/ElfnotesKewlTune/step-0400/
  base.flax_model
  chain-head.flax_model
```

**2ファイルが対になったモデルです。片方だけを更新・読み込みしてはいけません。** 基礎RNN方策のlogitsに、公開チェーン情報を受け取るheadの補正を加えます。モデルの上に中央配置の`PlacementPolicy`、さらに攻守表示の`PositionPolicy`を適用します。

| 指標 | 現在の記録と意味 |
| --- | --- |
| 検証用の展開 | 65/66。現在の保持検査の基準 |
| 過去の最終試験 | 86/88。凍結したニューラル方策の当時の記録。今回再測定していない |
| 補助を含む別の回帰試験 | 44/44。88初手の新しい独立試験という意味ではない |
| 実戦の強さ | 上の数字だけでは判断できない |

接続済みの内容：

- 自己対戦と対戦評価で、同じbase＋headと配置・表示形式の補助を使用。
- baseとheadの両方を学習・保存・読み込みする。
- 配置・表示形式のプロンプト（msg 11 / 5）は外部補助が実行する。現在は該当行をPPO損失のmaskから外している。
- RNN状態と補助の記憶を環境・実プレイヤーごとに扱い、対戦リセットに対応。
- `scripts/continuous_opening_policy.py`は`state.json`の`best_checkpoint`を読み込む推論入口。呼び出しだけで学習は始まらない。
- 追加学習は重みからのwarm start。最適化状態・乱数・途中の対戦状態をすべて保存した完全再開ではない。
- 別チャットが原本を更新しても、自動的には混ぜない。各実験はモデル・語彙・エンジン・補助コードを固定する。

**自己対戦側には、先攻展開の手本を復習する保持用損失はまだありません。** 以前の教師あり展開学習にはCE＋KLがありましたが、それが現在のPPO自己対戦へ自動的に接続されているわけではありません。

## 5. なぜ変更を提案したか：実際に起きた退行

独立したテスト領域で、現在のモデルから256判断だけ自己対戦学習しました。

```text
/home/hatk8/projects/yugioh-ai/experiments/
continuous-opening-integration-check-20261109-v2/integration-result.json
```

- base・headの両方が変化し、重みはすべて有限。
- 既知の展開成功数：**65/66 → 57/66**。元が成功していた8件を失った。
- 8戦の対戦確認は2勝6敗。すべて自然決着、異常終了0。ただし少数なので実力差の結論には使わない。
- 保持検査が退行を検出し、試験候補は採用されなかった。
- 通常運転は起動せず、採用モデルのハッシュも変更していない。

失ったcase_id：

```text
cue-mix-jj-track-g08-p0
cue-reco-jj-zaren-g09-p0
mixed-rukina-cue-g08-p0
mixed-rukina-cue-g09-p0
mixed-rukina-cue-g10-p0
tune-juke-veiler-g08-p0
tune-juke-veiler-g09-p0
tune-stone-sweeper-g09-p0
```

これは保持対策を検討する根拠ですが、原因が破滅的忘却だけだと確定したわけではありません。更新量、報酬の疎さ、推論・学習経路の差、RNN処理等も調べる余地があります。

既存の保持検査は「悪化した重みの採用を防ぐ」仕組みです。「悪化しにくい学習方法」自体は別途必要です。

## 6. 研究の要点と、このプロジェクトへの適用範囲

### 6.1 主な参考研究：CLEAR

David Rolnickほか、**Experience Replay for Continual Learning**、NeurIPS 2019。

新しい経験と過去の経験を併用し、過去の方策・価値出力を保つ損失を加えることで、AtariとDMLabで忘却を減らした研究です。原手法はV-traceで古い経験との方策分布の差を補正し、方策のKLと価値の二乗誤差も使います。容量を制限したリプレイも実験しています。[要旨](https://arxiv.org/abs/1811.11682)・[本文の手法節](https://arxiv.org/html/1811.11682v2#S3)

**このAIに対する有効性や忘却ゼロを示した研究ではありません。** 以下の実装案は研究から着想した本プロジェクト用の提案です。PPOへBCの補助損失だけを追加する段階では「CLEARの完全実装」と呼ばないでください。

### 6.2 補足：PPOとの接続

John Schulmanほか、**Proximal Policy Optimization Algorithms**、2017。環境でのデータ収集と代理目的の最適化を交互に行い、ミニバッチで複数回更新する手法です。[原論文](https://arxiv.org/abs/1707.06347)

本プロジェクトでの実装判断：過去の教師データを、現在のPPO rolloutと同じ扱いで無条件に流し込まないでください。最初は新しいrolloutのPPO損失と、過去の手本に対する教師あり補助損失を分ける案を推奨します。過去データでRL更新まで行う完全なリプレイ方式は、別途off-policy補正を設計する次段階です。

## 7. 推奨変更A：自己対戦に、検証済みの展開の復習を追加する

### 7.1 最初の実装範囲（提案・未実装）

まず現在のPPO・エンジン・報酬を維持し、学習用の展開系列に対する補助損失を追加する小規模実験を作ります。実装場所の案は`/home/hatk8/projects/yugioh-ai/scripts/opening_rehearsal.py`等です。新規ファイル名・設定名はClaudeが最終設計してよいですが、学習ループからLLM/APIを呼ばない条件を守ってください。

概念式：

```text
L_total = L_PPO(fresh_rollouts)
        + lambda_bc * L_acceptable_action_CE(training_sequences)
        + lambda_kl * KL(frozen_teacher || current_policy)  # 任意の追加比較

L_acceptable_action_CE(s)
    = -log sum_{a in verified_acceptable_actions(s) intersect legal_actions(s)} p_current(a | history)
```

最初はBCの有無の比較を優先し、任意のKLは`lambda_kl=0`から切り分ける案です。KLを追加する場合、参照は実験開始時に固定した採用モデルとし、教師側へ勾配を流さず、参照モデルのハッシュを記録します。係数や復習頻度は論文由来の確定値ではありません。各損失の正規化・勾配の大きさ・処理時間を測り、小さな候補範囲で比較してください。

一つの操作列だけを厳密に固定する目的ではありません。同じ状態に対する検証済みの正解操作集合を尊重します。異なる状態で正しかった操作を単純に合併してはいけません。

### 7.2 利用できるデータと既存コード

現在の方針に対応した学習実験：

```text
/home/hatk8/projects/yugioh-ai/experiments/all-openings-bar-alternative-20261108/
run-20261108-194920/
  manifest.json
  dataset/ElfnotesKewlTune/train.npz
  dataset/ElfnotesKewlTune/validation.npz
  dataset/ElfnotesKewlTune/holdout.npz
  code/train_all_openings.py
  code/all_openings_common.py
  code/all_opening_cases.py
```

復習に使うのは**train.npzだけ**です。

| データ | 系列・判断数 | 用途 |
| --- | --- | --- |
| train | 416系列・25,802判断 | 復習学習に利用可能。現行ルール・観測・操作対応を確認する |
| validation | 78教師系列・4,848判断。自由実行では重複初手をまとめて66件 | モデル選択・回帰検査のみ。教師損失へ入れない |
| holdout | 88系列・5,130判断 | 過去の最終試験。反復選別・学習・係数調整へ入れない |

手札の並べ替えや同じ手札の別ルートも同じ区分として扱う既存の分割方針を維持します。既存88件はすでに成績が知られているため、今後の最終性能を確認する際には別の新しい固定試験も用意するのが望ましいです。

train.npzの配列は実ファイルで次を確認しました。先頭の2軸は系列416×最大長120です。

| キー | dtype | shape |
| --- | --- | --- |
| `cards_` | uint8 | 416×120×160×41 |
| `global_` | uint8 | 416×120×23 |
| `actions_` | uint8 | 416×120×128×12 |
| `h_actions_` | uint8 | 416×120×32×14 |
| `chain` | int64 | 416×120×4 |
| `labels` | int32 | 416×120 |
| `targets` | uint8 | 416×120×128 |
| `lengths` | int32 | 416 |
| `weights` | float32 | 416×120 |

`targets`は許容操作集合を表します。`labels`の単一操作だけを正解に戻さないでください。

既存の`code/train_all_openings.py`の`unroll()`・`loss_fn()`は実装の参考になります。ただし、**そのファイルはimport時に準備・学習のトップレベル処理へ進む構造**です。無条件にimportせず、必要な純粋関数を新しいモジュールへ切り出すなどしてください。古いmanifestの開始モデルや出力先をそのまま使ってはいけません。

### 7.3 学習経路で特に確認すること

- baseとheadを合成した最終logitsに復習損失をかけ、両方へ適切に勾配が届くこと。
- 法的に有効な操作mask、系列長、padding、`weights`を反映すること。有効な正解がゼロのデータは不整合として明示すること。
- 現行の配置・表示形式補助の責務を維持し、msg 5 / 11を方策の教師損失からも原則除外すること。補助を学習へ置き換える場合は別変更として扱うこと。
- 損失を付けないステップも、RNNの履歴として必要なら系列から落とさないこと。対戦境界・プレイヤーの扱いを保持すること。
- 系列先頭からのunroll、または妥当なburn-inを使うこと。ばらばらの判断を全部ゼロRNN状態で学習したり、古い重みで作った隠れ状態を無検証で使ったりしないこと。
- `chain`から実行時の`chain_`への対応、語彙、action index、合成モデルのparams構造を明示的に合わせること。
- `batch_stats`等がある場合、教師の状態を固定し、補助系列の更新で実戦側の状態を意図せず壊さないこと。
- 確認済みの観測は自分・公開情報だけ。相手の非公開手札、山札順、試験の目標盤面、手順番号を方策入力へ入れないこと。
- 召喚権の温存を合否・加点・減点へ戻さないこと。旧方針の手本を自動的に混ぜないこと。

## 8. 推奨変更B：妨害後の評価を広げる

### 8.1 現在はどう動くか

自己対戦学習はゲームエンジンの報酬を使います。指定の目標盤面を作れなかっただけで、その場で専用の失点を与える設計ではありません。

一方、採用前の展開テストは、既知の初手と指定妨害ごとの条件を調べます。すでに一部の別ルートや妥協盤面を受け入れていますが、未知の妨害から実現可能な最善盤面を自動で求める仕組みはありません。

具体例：現行の`rukina_recovery_witness.py::goal()`では、特定の初手・指定タイミングでルキナのサーチへうらら／ヴェーラーを受ける条件は、中央ストリチア＋横のルキナが目標です。必要なレベル1チューナーを初手に持つ条件です。獄神精の墓地サーチへうららを受ける条件には、クリスタルウィングを残す目標があります。これを全初手・全妨害への一般的な解答として扱わないでください。

### 8.2 評価変更の提案

| 層 | 確認するもの | 扱い |
| --- | --- | --- |
| 無妨害の基本展開 | 合法に成立する、既知の有効な着地点 | 既存の保持基準を残す |
| 指定された単発妨害 | 妨害が所定の状況で実際に使われたことと、その後の復旧 | 条件別に検証済みの複数の着地点を許容 |
| 未登録の妨害・複数妨害 | 選んだ行動、残った盤面・手札、後続、実戦成績 | 探索・診断として別集計。未知の結果を自動合格にしない |

「妨害を受けたら何でも合格」にせず、条件ごとの複数の着地点を定義します。到達可能性と有用性は別々に確認します。ルールエンジン上の合法な再現手順は到達の証拠になりますが、その盤面が強い証拠には対戦や具体的な妨害・後続の検査も必要です。

盤面の候補は、妨害として使える効果、維持したリソース、次のターンの動きなどを記録します。カード枚数やATKの単純な合計を、そのまま強さの正解にしないでください。これらの診断値を最初から学習報酬へ足すのも、別の検証を要する変更です。

現在未登録の妥協盤面は、元の判定・操作列・終局状態を残して「要確認」に分類できるようにする案です。**要確認の候補を自動採用しないこと。** 診断が増えても、既存の不合格を隠したり失敗件数から消したりしてはいけません。

指定妨害を受けてから復旧する能力の試験と、妨害を回避・踏ませる能力の試験は分けます。既存テストには妨害が実際に解決されたことを要求する条件があり、うまく回避した動きがその試験では不成立になる可能性があります。試験の目的を明示して別枠で評価します。

### 8.3 評価器を変更するときの条件

- 新しい合格条件は、合法な再現例と、届いていない例・効果を使えない例などの負例で確認する。
- 評価ルールに版を付け、元の採用モデルと新候補を**同じ新版**で再評価する。
- 旧版の65/66や86/88を、新版で測定した結果として書き換えない。
- 同じ候補を通すために、その場で閾値・目標を緩めない。条件変更と候補の比較を分ける。
- 新ルートを学習へ足す場合は学習区分から生成する。validationで見つけた成功操作列をtrainへ流用しない。
- 現在の「基準が解けたcase_idをすべて維持」という採用前検査は、最初の復習実験では維持する。複数の着地点を認めることと、この保護を丸ごと削除することを混同しない。

## 9. 小規模な比較実験の進め方（提案）

まず変更Aだけを比較し、評価器の変更Bは別段階にするのを推奨します。同時に変えると、保持が改善したのか採点が変わったのか分からなくなります。

1. 新しい独立した実験rootと設定を作る。原本・採用モデル・既存STOP・本番stateを保持する。
2. 開始するモデル2つ、補助、デッキ、語彙、エンジンを同一に固定し、学習前に65/66の基準を再現する。
3. A：現行PPO。B：同じPPO＋学習用展開の復習。最初はそれぞれ256判断程度で動作と退行を比較する。128の倍数という既存の学習単位を守る。
4. 動作確認後、例えば各4,096判断・複数seedの有限比較を別途計画する。これは提案値であり、現時点で実行済み・実行許可済みではない。
5. 保持率だけでなく、失ったcase_id・新しく解けたcase_id・家族別の成績・損失と勾配・所要時間・メモリ・Luaエラー・異常終了を保存する。
6. 展開維持を確認した候補について、先後を入れ替える同一seedペアで採用モデルと対戦する。将来は複数の固定相手への成績も確認する。
7. 候補の採用と常時運転の再開を分ける。有限試験の完了だけで通常運転を始めない。

同じseedを使っても、方策が変われば途中の訪問状態まで同じになるわけではありません。各条件の環境判断数と復習更新数・総計算時間は別々に記録します。既存256判断の記録は問題の再現証拠であり、新実装との公正なA/B比較を済ませた証拠ではありません。

**成功判定の第一段階**は、動作が正しいこと、学習区分を守ること、現在の基準で解けたケースを維持できることです。単一seed・少数対戦で強くなったと結論しないでください。維持できなければ保持の係数・更新頻度・経路整合性等を調べ、長時間運転へ進めません。

## 10. 変更箇所を探すためのコード案内

以下の相対パスはすべて`/home/hatk8/projects/yugioh-ai`からです。行番号は作成時点の目安です。

| ファイル・関数 | 役割・変更時の注意 |
| --- | --- |
| `scripts/continuous_learning.py::load_config()`（73行付近） | 仮想環境Pythonのパス保持、128単位の学習、対戦数の整除条件 |
| 同`runtime_check()`（122行付近） | native・語彙・デッキとsnapshot内のハッシュ照合 |
| 同`evaluate_candidate()`（353行付近） | 開始時に展開検査。通れば対戦、初期参照との比較、別seedの確認 |
| 同`run_loop()`（444行付近） | 学習・評価・採用・停止。展開検査で不合格ならlatestをbestへ戻す |
| `scripts/continuous_opening_backend.py::CompositeAgent`（84行付近） | baseとheadの合成、paramsとbatch_statsの扱い |
| 同`augmented_preprocess()`・`AdapterBank` | 公開チェーン特徴と補助担当mask、実行操作の補正・記憶 |
| 同`patched_trainer()`（199行付近） | cleanba.pyへ決まった場所で変更を挿入。文字列出現数の検査がある |
| 同`initialize()`（218行付近） | 固定snapshotを生成し、初期モデル2つを複製 |
| 同`train_cycle()`（259行付近） | snapshot/train.pyを有限の判断数で実行 |
| 同`openings_job()`（293行付近） | validationの自由実行。初手をまとめて66件。holdoutを使わない |
| 同`regression_ok()`・`regression_gate()`（315行付近） | case_id単位の保持。件数だけ同じでも入れ替わりを認めない |
| 同`install_evaluation()`（374行付近） | 評価器に合成モデルと同じ補助を接続 |
| `scripts/continuous_opening_policy.py` | 採用モデルを読み込む推論専用入口 |
| `config/continuous-elfnotes.json` | 現行の本番向け設定。初回比較ではコピーし別rootで使う |
| `env/continuous-opening-runtime.json` | 使用するnative・語彙・デッキの対応とハッシュ |
| `tests/test_continuous_learning.py` | 共通の停止・復旧・採用条件などの回帰検査 |
| `tests/test_continuous_opening_backend.py` | 2モデルの保持、補助の学習mask、case_id維持・復帰等 |
| `scripts/evaluate.py` | レビュー済みの対戦評価器。今回の保持損失追加で無関係に変更しない |

評価の実行コードは、現在rootの`/snapshot/policy-code/all_opening_witness.py`と`rukina_recovery_witness.py`です。全体の召喚権方針は`docs/evaluation-policy-current.md`、現行方針へ移した評価コードは`evaluation/normal-policy-v2/code/`と`scripts/eval-combo-benchmark.py`です。どのコードを実際にimportしているか確認してから変更してください。

### 特に注意：固定snapshotはソース変更だけでは更新されない

現在のrootはすでに初期化済みです。`scripts/continuous_opening_backend.py`を変更しても、既存の`snapshot/train.py`や`snapshot/policy-code`が自動生成し直されるわけではありません。新しい実験rootで新しいsnapshotを作り、使用したソースとハッシュを記録してください。

旧snapshotを直接編集してハッシュだけ合わせ、過去の実験を新コードで行ったことにしないでください。後日本番へ反映する場合も、版の分かる移行手順が必要です。

現行の対戦選別は、採用モデルと128戦、初期モデルと64戦、有望な場合は別seedで256戦です。確認済みで異常なし、ペア単位95%区間の下限>0.5等を使います。これは運用上の条件で、無限に繰り返す候補比較全体の誤採用率を保証しません。

展開不合格時のlatest→bestへの復帰は実装されています。一方、展開は通るが対戦で昇格しない候補については、latestから次の学習へ進む場合があります。推論入口はbestです。「未採用なら常に全学習状態が初期値に戻る」と説明しないでください。

## 11. 必要な検証・完了時に残すもの

既存の11項目は前回の接続作業で合格しています。この資料作成ではコードを変更していないため再実行していません。Claudeが変更した後は、仮想環境で適切に再実行してください。

```bash
cd /home/hatk8/projects/yugioh-ai
/opt/ygo-ai/.venv/bin/python -B -m unittest discover -s tests -p 'test_continuous*.py' -v
```

追加する検証は、実際の不具合を防ぐ内容にしてください。

- 複数の許容操作を正しく扱い、違法操作・padding・補助担当操作が損失へ入らない。
- train以外を復習へ指定すると明示的に拒否し、手札単位の区分とデータハッシュを確認する。
- 系列境界でRNNをリセットし、チェーン特徴・base＋headの合成が推論と一致する。
- 補助損失有効時にbase/headへ期待する勾配が届き、凍結教師が変化しない。無効時には既存経路と整合する。
- 展開失敗時に未採用となること、latest/bestの整合、2モデル保存・読込、STOPの維持を確認する。
- 評価器を拡張する段階では、正例・負例・指定妨害の発動／解決・未登録結果の保留・旧版記録の保持を確認する。
- 独立したrootで実際の短い学習・自由展開・対戦を行い、ソースだけでなく生成されたsnapshotが使われた証拠を残す。

完了報告には、変更差分、実行コマンド、設定・seed・各判断数、復習対象のmanifest・ハッシュ、モデル2つのハッシュ、検証用のケース別結果、対戦の確認済み数と除外理由、時間・負荷、未解決点、通常運転が停止のままであることを含めてください。

タイムアウト確定・疑いは、同じseedの先後ペアを確認済み集計から外す既存仕様を維持します。打ち切りを含む総合成績と自然決着のみの成績も区別します。対戦数を減らして合格しやすくする変更はしません。

## 12. 追跡に使う資料・ハッシュ

最初に読む資料（プロジェクト内）：

1. `/home/hatk8/projects/yugioh-ai/docs/evaluation-policy-current.md`
2. `/home/hatk8/projects/yugioh-ai/docs/continuous-opening-integration-2026-11-09.md`
3. `/home/hatk8/projects/yugioh-ai/config/latest-verified-opening-policy.json`
4. `/home/hatk8/projects/yugioh-ai/docs/all-openings-training-2026-11-08.md`
5. `/home/hatk8/projects/yugioh-ai/config/continuous-elfnotes.json`
6. 現在rootの`state.json`、`opening-baseline.json`、`snapshot/hashes.json`、`snapshot/opening-manifest.json`
7. テスト領域の`integration-result.json`と`cycles/smoke-256/opening-regression.json`

モデル・環境の固定値：

| 対象 | SHA256 |
| --- | --- |
| 採用初期base | `966cc8b1c3f0c092e824cc52dffcaf9c71f0c9373ece0fd07831d4a9df4f7cd1` |
| 採用初期head | `1aaf569daf53a4453ef2f2e450c924af014aea83b3e4f98fea2644b795b52dc8` |
| native | `19a60f38a3d39b61a2ee082168d5a788a1022b2eb569633816596c8b3de21294` |
| 語彙 | `aeb5be89d4a5591ccc5cce6af96def74ef47463a55265f75c11b92029e772e7b` |
| デッキ | `6e6344d98ee5865e8ea1460c75c361a3f6e054c02cb8a041337c532646d15373` |
| 復習候補train.npz（manifest記録） | `0ef4700c92ddb6f26c3e3f5dc04d4893d2cb02c096304af0bbdc546f8070e4da` |

作成時点で読み直したコード・設定のSHA256：

| 対象 | SHA256 |
| --- | --- |
| `scripts/continuous_learning.py` | `27ebfb29049bf8f9b146fbb823dc3a599fea59c46a758efc2ea07f2dc9d6da2a` |
| `scripts/continuous_opening_backend.py` | `3789adca37334d8a9550a5ca7b2d044d6fc92c4db860534694d8155377845f6e` |
| `scripts/continuous_opening_policy.py` | `8f6ef2904e301261dff0ee25bd515678d311f62473d63d9754e228ecf0ca295e` |
| `scripts/evaluate.py` | `67db15ffeb9f94884dda433352f52170ac0b7d60de46056c88f0317177d9d7d0` |
| `tests/test_continuous_learning.py` | `8a7cc09fbb90dad9b5f326f95b08022baaf5c909425e396753deb17c6e717fba` |
| `tests/test_continuous_opening_backend.py` | `c677877a827b2e065aae1971c88e620bc9c77855c05610669376737dc8cd63c9` |
| `config/continuous-elfnotes.json` | `5a6fab46def00dce6670310b2b807efb7a9b9b0679b0f411934defb1a5c6c26e` |

資料作成後に別チャットが変更する可能性があります。ハッシュが違えば、その差分を確認してから変更してください。値を機械的に過去へ戻すための表ではありません。

## 13. Claudeへ渡す短い依頼文

以下はユーザーが後日、コード変更を依頼するときに貼り付けるための文面です。この資料を作成した段階では、下記の作業はまだ実行していません。

```text
添付の「遊戯王AI：研究メモとClaudeへのコード変更引き継ぎ」を読み、現在の実ファイルを確認してから変更してください。

まず、現行の自己対戦PPOへ、現行方針のtrain.npzだけを使う展開復習の補助損失を追加し、復習あり・なしの有限な比較試験を独立した実験rootで行ってください。モデルはbase＋chain-headの組、配置・攻守表示補助、公開情報だけの観測、通常召喚を採点しない方針を維持してください。

最初は資料の変更Aを対象とし、評価器の変更Bは別の段階として設計を残してください。既知のcase_idを維持する採用前検査は外さないでください。validation・holdoutを学習に使わず、現在の採用モデル・実験記録を保護してください。

通常運転は停止したままです。本番のSTOPを削除せず、常駐・自動起動を再開しないでください。小規模試験の条件と上限を記録し、結果に基づいて変更点・検証・限界を報告してください。学習ループからLLM/APIを呼ぶ仕組みは追加しないでください。
```
