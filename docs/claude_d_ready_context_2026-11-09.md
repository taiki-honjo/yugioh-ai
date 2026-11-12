# Claudeへの引き継ぎ：採用済み16,384判断モデルをD条件で学習再開できるよう準備する

作成日：2026-11-09（日本時間）。この日の実ファイルを読み直したコンテクスト。

**今回の依頼は「D条件での再開準備」まで。通常運転・自動起動の再開、追加学習、新候補の採用は行わない。** この資料の作成時点では、設定・コード・採用モデルを変更していない。

## 1. ユーザーの目的

- 自分のPCのWSL環境で、エルフェンノーツ＋キラーチューンのAIを学習・改善したい。
- 通常の学習はPython等のコードだけで完結させ、Codex・Claude・LLM/APIを学習ループから呼ばない。
- 今後のコード変更はClaudeへ依頼する。
- 通常運転の停止指示は継続中。今回の準備完了だけで起動してはいけない。

今回解消する問題：**採用モデルはD条件で学習した成果へ切り替わったが、将来の学習用設定はまだ「学習率3e-5・復習なし」であり、固定された本番snapshotも復習なしのまま。** モデルの切り替えと、学習方法の切り替えは別である。

## 2. 環境と現在の参照先

| 項目 | パス・値 |
| --- | --- |
| WSL | Ubuntu-22.04、ユーザーhatk8 |
| プロジェクト | `/home/hatk8/projects/yugioh-ai` |
| Windowsからの参照 | `\\wsl.localhost\Ubuntu-22.04\home\hatk8\projects\yugioh-ai` |
| Python | `/opt/ygo-ai/.venv/bin/python` |
| デッキ | `/home/hatk8/projects/yugioh-ai/assets/deck/ElfnotesKewlTune.ydk` |
| 本番設定 | `/home/hatk8/projects/yugioh-ai/config/continuous-elfnotes.json` |
| 現本番root | `/home/hatk8/projects/yugioh-ai/experiments/continuous-elfnotes-openings-d16384-20261109` |
| 現方針設定 | `/home/hatk8/projects/yugioh-ai/config/verified-opening-policy-d16384-20261109.json` |
| 採用モデル原本 | `/home/hatk8/projects/yugioh-ai/experiments/adoption-d16384-20261109/model/` |

Python実行ファイルの最終シンボリックリンクを`resolve()`してシステムPythonに変えないこと。仮想環境の実行パスを維持する。

確認時、プロジェクトとその親・docs内に適用する`AGENTS.md`は見つからなかった。以前確認した時点ではプロジェクト直下はGitリポジトリではなかったため、変更前のファイルとハッシュを別途保存して差分を残す。後日の変更があれば実ファイルを優先する。

## 3. 採用済みモデルと実績

2026-11-09 16:40に、累計16,384環境判断の候補を**対戦性能を優先した手動採用**として反映済み。

| 対象 | SHA256 |
| --- | --- |
| 採用base | `86b4358dcfec696cb4b3cc4c19781c34292b2509bf037b9e4f44d9ed8ec59d1c` |
| 採用chain-head | `6ca3221f1828eecc5cea817003c8df5e8802b75fabef0f4aa33105ad0f979fce` |
| 配置補助v2 | `3e96241095487fb48426099d774b7813cee348fc282dacbde0ced9af2d7abbba` |
| 旧採用base | `966cc8b1c3f0c092e824cc52dffcaf9c71f0c9373ece0fd07831d4a9df4f7cd1` |
| 旧採用chain-head | `1aaf569daf53a4453ef2f2e450c924af014aea83b3e4f98fea2644b795b52dc8` |

baseとheadは必ず組として扱う。配置補助v2と攻守表示補助も両方の推論入口で同一であることを維持する。

| 評価 | 記録 |
| --- | --- |
| 既存66件 | 65/66、旧採用モデルが成功したケースの喪失0 |
| 固定試験88件 | 87/88。`jj-cue-rotary-ft00`を喪失し、`jj-cue-reco-ft03`が新たに成功 |
| 旧採用モデルとの128戦 | 82勝46敗、ペア95%区間[0.511, 0.757]、seed 80016384 |
| 別seedの確認256戦 | 170勝86敗、ペア95%区間[0.575, 0.745]、seed 80016385。256戦単独で対戦基準を満たした |

固定試験の保持条件は不合格だった。その記録は消さず、今回の手動採用理由として明記されている。128戦と256戦は合算して判定していない。確認できた対戦性能の改善は、旧採用モデルとの同デッキ対戦の範囲である。

今回、採用をやり直したり、旧モデルへ戻したり、保持不合格の履歴を書き換えたりしない。今回の手動採用を、今後の自動採用基準を無条件に緩める理由にしない。

## 4. 現在の停止状態と、未反映の学習設定

本資料作成時の現本番root：

```json
{
  "status": "stopped",
  "phase": "idle",
  "pid": null,
  "pending": null,
  "completed_cycles": 0,
  "added_steps": 0,
  "latest_checkpoint": "/home/hatk8/projects/yugioh-ai/experiments/continuous-elfnotes-openings-d16384-20261109/models/initial",
  "best_checkpoint": "/home/hatk8/projects/yugioh-ai/experiments/continuous-elfnotes-openings-d16384-20261109/models/initial"
}
```

`STOP`は存在し、内容は`now`。このrootの`models/initial`は、旧モデルではなく、新しく採用した16,384判断モデルの複製である。

本番設定で実際に確認した値：

```json
{
  "backend": "verified_opening",
  "learning_rate": 0.00003,
  "bootstrap_steps": 4096,
  "steps_per_cycle": 65536,
  "eval_envs": 4,
  "eval_episodes": 128,
  "reference_episodes": 64,
  "confirmation_episodes": 256,
  "max_eval_steps": 4000,
  "job_memory_gib": 5,
  "cpu_percent": 300,
  "job_timeout_seconds": 1800,
  "rest_seconds": 60,
  "keep_cycle_checkpoints": 8,
  "minimum_free_gib": 8,
  "max_consecutive_failures": 3,
  "opening_regression_required": true
}
```

`rehearsal`は設定されていない。さらに設定の`initial_checkpoint`は以前のstep-0400を指したままだが、verified-opening backendの初期化では`verified_policy_config.selected_checkpoint`が実際の開始点であり、新採用モデルが読み込まれている。新しい設定では、開始点の表記も新採用モデルと矛盾しないように整える。

`initial_training_steps=69632`等は旧RL起源の履歴であり、今回の新rootで追加した判断数ではない。教師あり展開学習の更新数と、累計16,384の追加環境判断、今後新rootで追加する判断数を混同しない。

## 5. 今回準備するD条件

過去に実際に使った設定：

```json
{
  "learning_rate": 0.000015,
  "rehearsal": {
    "enabled": true,
    "split": "train",
    "lambda_bc": 5.0,
    "lambda_kl": 0.0,
    "sequences_per_minibatch": 2,
    "seed": 20261109,
    "sha256": "0ef4700c92ddb6f26c3e3f5dc04d4893d2cb02c096304af0bbdc546f8070e4da"
  }
}
```

これはPPOの新しいrolloutに対する勾配と、学習用の成功展開に対する許容操作集合CEの勾配を足す方式。CLEARの完全実装ではない。KL・価値保持・過去データによるRL更新を今回新たに追加しない。

復習データ：

```text
/home/hatk8/projects/yugioh-ai/experiments/all-openings-bar-alternative-20261108/
run-20261108-194920/manifest.json

/home/hatk8/projects/yugioh-ai/experiments/all-openings-bar-alternative-20261108/
run-20261108-194920/dataset/ElfnotesKewlTune/train.npz
```

416系列・25,802判断。損失対象は19,178判断。`train-witnesses.json`と合わせて固定し、manifest・データのハッシュと初手区分の分離を確認する。validation・holdout・固定試験88件の操作列を学習へ混ぜない。

現在の65,536判断／cycleは、その長さでD条件の保持・対戦改善を確認した値ではない。再開用の準備では、**bootstrapと1cycleをともに4,096判断とすることを提案する**。これは実行上限を小さく保つための設定で、起動を許可するものではない。長い区間への変更や通常運転の開始は後の依頼に分ける。

既存の資源上限・空き容量確認・停止処理を維持する。実験用の全checkpoint永久保持を、常時運転向けに無条件で有効化しない。

## 6. 実施してほしい作業

1. 現本番設定、方針設定、state、STOP、基準結果、snapshot、モデル2つのハッシュを記録して控えを作る。
2. D条件用に新しい本番向けrootを作る。最初に`STOP=now`を置く。名前の例は`experiments/continuous-elfnotes-openings-d16384-rehearsal-ready-20261109`。既存rootを上書きしない。
3. 新採用モデルのbase＋headをそのまま開始点にする。方針設定・state・モデルハッシュを照合し、重みを変更しない。
4. 学習率1.5e-5と上記の復習設定で、trainer・復習モジュール・trainデータ・補助・評価器を新snapshotへ固定する。D条件がconfigだけでなく、実際のtrainerに接続されていることを確認する。
5. 後述の「本番は復習なし」という前提を持つ実験ランナー／テストを、D条件の本番にも対応できるよう修正する。
6. 新rootでのモデル読み込み、65/66の基準とcase_id集合、推論入口、runtime/snapshotのハッシュ検査、停止状態を確認する。学習はしない。
7. 検証後、本番設定を新しい停止中のrootとD条件へ切り替える。採用モデル自体と配置補助の版は変えない。旧設定1つを戻せる切り戻し手順を残す。
8. 最後に、新旧rootのSTOP・停止状態・プロセスを確認し、準備完了として報告する。

既存snapshotを直接書き換えてハッシュを合わせる方法は使わない。`initialize()`は既存stateがあるrootを再生成しないので、同じrootのconfigへ`rehearsal`を追加するだけでは不十分である。

**実施しないこと：** `resume`、`run`、`run --once`、WindowsのStart操作、追加学習、対戦評価、候補の選別・新規採用、ロタリー等のカード処理変更、採点方針変更。

`run --once`も学習を行うため、準備確認の代わりに使用しない。モデルを更新しない有限の推論・基準展開検査と、単体テストは実施してよい。

## 7. ソースで確認した対応点

全パスは`/home/hatk8/projects/yugioh-ai`から。

### 7.1 新snapshotへの復習接続はすでに存在する

`scripts/continuous_opening_backend.py::initialize()`（237行付近）は、`cfg.rehearsal.enabled`が有効なら`opening_rehearsal.prepare_snapshot()`で新しいtrainerを構成する。

`train_cycle()`（288行付近）は、`cfg.learning_rate`を実際の`--learning-rate`へ渡す。ここを無関係な新トレーナーへ置き換える必要はない。

`scripts/opening_rehearsal.py`はtrain限定のハッシュ照合・系列unroll・許容操作集合CE・PPOとの勾配合成を実装済み。配置・表示形式（msg 11/5）とpaddingは損失から外す。複数の許容操作とRNN履歴を維持する。

### 7.2 実験ランナーは「本番trainer＝復習なし」を仮定している

`scripts/opening_rehearsal_ab.py::check_snapshot()`（193行付近）：

```python
identical_trainer = ours['train.py'] == theirs['train.py']
if bool(arm.get('rehearsal')) == identical_trainer:
    raise RuntimeError('Trainer identity does not match the arm definition')
```

本番が復習ありになると、同じD条件のarmはtrainerが同一でも拒否され、復習なし対照も本番と異なるため拒否される。**本番との単純な同一／非同一で判定せず、それぞれの実効設定から期待されるtrainerを構成・照合する等の方式に直す必要がある。**

ガードを削除して何でも許す変更にはしない。学習率・復習設定・checkpoint保存設定・開始モデル・コード版・データ版・コマンド照合を維持する。

`build_arm_config()`は、base configの`rehearsal`を一度外し、armに明示されたものを入れ直している。対照armの「復習なし」と、設定省略時の扱いを明示し、D条件が意図せず対照へ混ざらないことをテストする。

### 7.3 テストにも同じ前提がある

`tests/test_opening_rehearsal.py::WiringTests.test_disabled_path_is_byte_identical_and_enabled_path_is_hooked()`（205行付近）は、復習なしで作ったtrainerが本番snapshotと一致することを要求する。

本番D化後はこの前提が成立しない。復習なし経路の保持と、有効な設定に対応したtrainerの一致を、それぞれ適切な比較先で検証するよう直す。期待値を単に新しいハッシュへ置き換えるだけで済ませない。

### 7.4 停止と再開の意味

現行の追加学習は、区間ごとに重みから開始するwarm startで、optimizer・RNG・対戦途中の状態は再起動する。この性質は今回変更しない。

実験・集計は、本番の指紋変更を検出して旧rootでの再利用を拒否する設計。今回の意図した本番設定変更も記録し、旧実験を再利用するために履歴・条件記録を書き換えない。

## 8. 維持する評価とモデルの扱い

- 今後の66件の保持基準は、新採用モデルで測った65件の成功case_id集合とする。件数だけの一致ではなくケース単位で維持を確認する。
- 固定試験88件は、新採用モデルの87件と旧モデルとの入れ替わりを区別して残す。今回の手動採用履歴も保持する。
- 現行の常時学習の自動回帰gateは66件を使う。固定88件が自動gateへ接続済みだと説明しない。今回の準備で無断で採用基準を増減しない。
- 対戦選別は、現在のレビュー済みの先後ペア評価と別seed確認を維持する。今回対戦は実施しない。
- 全候補の失敗・タイムアウト・打ち切り・評価不能を隠さない。snapshotとコードの版を混ぜない。
- 通常召喚・セットの回数を合否・加点・減点に使用しない。
- 入力は自分と公開情報だけ。相手の非公開手札や山札順、教師の手順・目標を渡さない。

ロタリーの手札確認は、効果で公開されたカードをモデル観測へ渡すところまで実装・確認済み。情報が消えた後の活用はRNN等に依存する。専用記憶や対策学習の追加は、このD条件準備の範囲に含めない。

## 9. 推論入口と停止中のWindows側

推論入口は以下の2つ。

```text
/home/hatk8/projects/yugioh-ai/scripts/continuous_opening_policy.py
/home/hatk8/projects/yugioh-ai/scripts/verified_opening_policy.py
```

両方が、現本番設定の参照先から同じ採用モデルとv2補助を読み込む。`verified_opening_policy.py`は`deck_policy.py`経由でも使われる。新しいプロセスで両方を検証し、別版のimport済み補助がある場合の拒否を維持する。

Windows側の既知のコントローラー：

```text
C:\Users\hatk8\AppData\Local\YugiohAI\continuous-elfnotes.ps1
```

ログオン時の入口はStartupショートカット。今回はWindows側や自動起動を変更しない。WSLから見えない場合は未確認と記載し、確認済みと報告しない。新rootのSTOPを最初に作り、最後まで維持する。

## 10. 検証と完了条件

前回の採用報告は65テスト合格だった。この資料作成時にはコード変更をしていないため、再実行はしていない。改修後は関連テストを実行する。

```bash
cd /home/hatk8/projects/yugioh-ai
/opt/ygo-ai/.venv/bin/python -B -m unittest \
  tests.test_continuous_learning \
  tests.test_continuous_opening_backend \
  tests.test_opening_rehearsal \
  tests.test_opening_rehearsal_runner \
  tests.test_center_placement_remix \
  tests.test_policy_entries -v
```

今回追加・更新するテストの焦点：

1. 本番が復習ありでも、復習ありarmと復習なし対照が正しいtrainerで検証される。不正な差分は拒否する。
2. 実効学習率1.5e-5・λ=5・trainデータのハッシュが、新config・snapshot・実行予定コマンド・条件記録で一致する。
3. 学習率や復習設定の変更後に、古い結果を同じ条件として再利用・集計しない。
4. 新rootの開始モデル2つは、採用済み16,384判断モデルと一致する。準備中に重みが変わらない。
5. 新旧rootでSTOPと停止状態が維持され、途中の失敗でも子プロセスが残らない。
6. 両推論入口の実パラメータ・補助の読み込み元が一致し、基準65/66の成功case_id集合を再現する。

完了報告に含めるもの：変更差分・変更前の控え・新rootとsnapshotのハッシュ・採用モデル2つのハッシュ・D設定・trainデータのハッシュ・基準結果・テスト結果・停止状態・切り戻し手順。学習を開始していないことを明記する。

「再開準備完了」は、今回のD条件の設定・接続・停止状態を確認できた意味。長期間の無人運転や、全seedでの保持・勝率向上を実証した意味ではない。

## 11. 最初に読む既存資料

全パスは`/home/hatk8/projects/yugioh-ai`から。

1. `docs/adoption-d16384-2026-11-09.md`
2. `config/continuous-elfnotes.json`
3. `config/verified-opening-policy-d16384-20261109.json`
4. `experiments/adoption-d16384-20261109/switch-record.json`、`root-check.json`、`entry-check.json`
5. `docs/opening-d16384-2026-11-09.md`
6. `experiments/opening-d16384-20261109/plan.json`と`summary.json`
7. `docs/evaluation-policy-current.md`
8. 現本番rootの`state.json`、`opening-baseline.json`、`snapshot/hashes.json`

現在のファイルのSHA256（後日変わっていれば差分を確認するための値）：

| ファイル | SHA256 |
| --- | --- |
| `config/continuous-elfnotes.json` | `b50810a34352f00a3f14f4bb79154f2c41b60e3d918c512aeeb23c8152b07d15` |
| `config/verified-opening-policy-d16384-20261109.json` | `d00f0473872daf3aa5b4ec6265d7ba38b2d3573b6dd4b71ba9dbb3da818e2611` |
| `scripts/continuous_learning.py` | `27ebfb29049bf8f9b146fbb823dc3a599fea59c46a758efc2ea07f2dc9d6da2a` |
| `scripts/continuous_opening_backend.py` | `17715b93f1976f2b7fd424a4a0a80668278f35e7a3fa714401570d915dadf0e3` |
| `scripts/opening_rehearsal.py` | `6aba1a3da68bfe5921e8e5bfc8146b6ba0273467a4dacd07209d579e7d760f56` |
| `scripts/opening_rehearsal_ab.py` | `aed3715ffadf3a1edbb2dab2d9f401b92f37d21a6516259a95cb5d54b92b4aa1` |
| `scripts/verified_opening_policy.py` | `22451de8d0ad65dc524e79195690679c100a982615cbdc7d2875e0dcd34d565a` |
| `scripts/continuous_opening_policy.py` | `8f6ef2904e301261dff0ee25bd515678d311f62473d63d9754e228ecf0ca295e` |

## 12. Claudeへ貼る依頼文

```text
添付の「採用済み16,384判断モデルをD条件で学習再開できるよう準備する」を読み、実ファイルを確認して変更してください。

今回の範囲は、通常運転を停止したままD条件の再開準備を完了することです。新採用モデル（base 86b4358d…／head 6ca3221f…）と配置補助v2を維持し、学習率1.5e-5・train限定の復習λ=5を、新しいrootとsnapshotへ固定してください。bootstrapとcycleは4,096判断の設定で準備してください。

本番trainerが復習なしだと仮定しているランナーのcheck_snapshotと関連テストを、実効設定に基づく照合へ対応させてください。データ分離、ハッシュ・コマンド照合、停止処理、既存の自動採用基準を維持してください。過去の実験・保持不合格・手動採用の記録は書き換えないでください。

新rootには他の作業より先にSTOP=nowを置いてください。検証後に本番設定を新しい停止中のD用rootへ切り替え、両推論入口・モデルハッシュ・基準65/66を確認し、切り戻し手順を残してください。

追加学習、対戦、resume/run/run --once、常駐・自動起動の再開はしないでください。学習ループからLLM/APIを呼ぶ仕組みは追加しないでください。テストと学習なしの有限検証を行い、差分・ハッシュ・設定・停止状態を報告してください。
```
