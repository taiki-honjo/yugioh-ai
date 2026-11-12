# 2モデル形式での基準モデル参照先の修正

2026年11月9日（日本時間）。依頼資料：`docs/claude_initial_reference_fix_2026-11-09.md`。

**修正した。** 次のことは行っていない。
- 学習・対戦・再開
- 候補の採用
- 評価の迂回

5回目の候補、採用モデル、本番設定、snapshot、`STOP=now`、`pending.phase=evaluating` は維持した。

## 1. 不具合と修正

`scripts/continuous_learning.py::evaluate_candidate()` の不具合と修正は次のとおり。

- **不具合：** 候補対bestの128戦の後、初期モデルとの参照対戦（64戦）で、相手に `models/initial.flax_model` を渡していた。2モデル形式（`verified_opening`）のrootには、`models/initial/`（base＋chain-head）しかない。`evaluate_job()` はこのディレクトリに `base.flax_model` を付けるため、存在しない `models/initial.flax_model/base.flax_model` を読もうとして失敗する。
- **修正：** `verified_opening` は `models/initial`（ディレクトリ）、従来の単体モデルbackendは `models/initial.flax_model` を渡す。`evaluate_job()` がディレクトリへ `base.flax_model` を付ける前提は変えていない。
- **変えていないもの：**
  - 判定基準、試合数（128／64／256）、seed（確認は別seed）、重み
  - 欠けた旧形式のファイルを作る回避はしていない。参照評価を省いたり、相手を置き換えたりもしていない。

```diff
-    command[command.index('--opponent') + 1] = str(Path(cfg['root']) / 'models/initial.flax_model')
+    reference_name = 'initial' if cfg.get('backend') == 'verified_opening' else 'initial.flax_model'
+    command[command.index('--opponent') + 1] = str(Path(cfg['root']) / 'models' / reference_name)
```

`continuous_learning.py`：`27ebfb29…` → `d9f6b303…`。このファイルはroot内のsnapshotに含まれないので、snapshotは作り直していない。

## 2. 回帰テスト（`tests/test_evaluate_candidate_reference.py`、`18b389af…`）

一時rootで、実際の `evaluate_candidate()` を通す。差し替えたのは、ジョブを起動する部分だけである。

各対戦ジョブは、同じプロセスの中で実物の次の処理を通す。
- `evaluate_job()`（2モデル形式のパス変換）
- 2モデル形式の読み込み処理（`install_evaluation` の `load`：`<dir>/base.flax_model` と `chain-head.flax_model` を読む）
- `scripts/evaluate.py` の `resolve_agent_spec()`（ファイルが無ければ例外）

対局はせず、決めておいた集計を返す。各モデルに異なる目印の値を入れ、実際に読み込まれたbase・headを照合する。

| テスト | 確認すること | 修正前 | 修正後 |
| --- | --- | --- | --- |
| 2モデル：参照対戦が初期のbase＋headを読む | 展開検査→候補対best→初期参照→確認評価。参照の相手は実在する `models/initial`。初期のbase・head、候補のbase・headが読まれ、候補対bestと確認評価の相手はbestである | ERROR | ok |
| 2モデル：試合数とseed | 128／64／256戦。seedはeval_seed、eval_seed、eval_seed＋1 | ERROR | ok |
| 2モデル：採用条件を迂回しない（4条件） | 候補対bestの信頼区間の下限が0.5以下、参照の勝率が0.5未満、参照が不完全のいずれかなら、確認評価に進まない。確認評価の結果で採用が決まる | ERROR×4 | ok |
| 2モデル：展開の喪失は対戦前に止まる | 1件喪失で、対戦ジョブ0件、不採用 | ok | ok |
| 単体モデル形式は従来どおり | 参照の相手は実在する `models/initial.flax_model`、試合数とseedも同じ | ok | ok |

- 修正前のエラーはすべて、本番で起きるのと同じ `FileNotFoundError: …/models/initial.flax_model/base.flax_model` だった。
- 関連する7ファイルをまとめて実行し、**75項目すべて合格**した（`experiments/initial-reference-fix-20261109/test-all-after-fix.txt`）。

### 既存テスト1件の調整

`tests/test_continuous_opening_backend.py::test_production_config_snapshot_and_command_agree` を調整した。

- **失敗の理由：** このテストは、「本番のbestとlatestが両方とも採用モデル」であることを求めていた。これは学習前だけ成り立つ条件である。19:16〜19:33の運転後、latestは評価途中の5回目の候補を指しているため、失敗した。
- **修正と無関係であることの確認：** 修正前のコードでも同じく失敗することを確かめた。
- **調整後の内容：**
  - rootの `models/initial` は採用モデルである。
  - bestは `models/initial` のまま、または採用されたなら `opening-best.json` がある。
  - latestは、bestか、現在のpendingの区間フォルダ内にある。

## 3. 維持したもの（`before.sha256`、153ファイル）

作業後に照合すると、変わったのは `scripts/continuous_learning.py` だけだった。

| 対象 | 値 |
| --- | --- |
| 5回目の候補 | base `b6aaa425…`、chain-head は作業前と同じ |
| 採用モデル | base `86b4358d…`、chain-head `6ca3221f…` |
| 本番設定 | `7ba275dc…` |
| snapshot | `hashes.json` `d4c135a9…` |
| root | STOP=now、stopped、pid・child_pidなし、completed_cycles 4、cycle 5 |
| pending | phase evaluating、steps 4,096、training_seed 10000005、eval_seed 80004985 |
| best / latest | best＝`models/initial`、latest＝`cycles/000005-715e84c3/candidate` |

- 学習・評価のプロセスはない。
- 5回目の途中で止めた評価は流用しない。再開すると、テストと対戦が最初からやり直される。

## 4. 注意

- 過去の実験ランナー（`opening_rehearsal_ab.py`）は、`continuous_learning.py` のハッシュを条件の指紋に含めている。このため、過去の実験の集計を再実行すると、指紋の不一致で拒否される。旧記録のハッシュは書き換えていない。
- この修正で確認したのは、参照対戦が正しいモデルを読み込むことまでである。実際の対戦は行っていない。

記録：`experiments/initial-reference-fix-20261109/`
- `source-before/`：修正前のソース・テスト・state・5回目の評価ログ
- `before.sha256`
- `continuous_learning.diff`、`test_continuous_opening_backend.diff`
- `test-before-fix-summary.txt`、`test-after-fix-summary.txt`、`test-all-after-fix.txt`
- `fix-record.json`
