# Claudeへの修正依頼：2モデル形式の基準モデル参照先

2026-11-09（日本時間）。実コードと現本番rootを読み直して確認。資料作成時点ではコード修正・学習・対戦・再開を行っていない。

## 今回の範囲

学習と対戦を停止したまま、参照先の不具合と回帰テストだけを修正する。5回目の候補・本番設定・採用モデル・pending・STOPを維持し、再開はしない。

## 確認した不具合

プロジェクト：`/home/hatk8/projects/yugioh-ai`

`scripts/continuous_learning.py::evaluate_candidate()`の374行付近：

```python
command[command.index('--opponent') + 1] = str(Path(cfg['root']) / 'models/initial.flax_model')
```

現本番rootには、次の組だけがある。

```text
models/initial/base.flax_model
models/initial/chain-head.flax_model
```

`models/initial.flax_model`は存在しない。

さらに`evaluate_job()`の342～343行付近は、`backend == 'verified_opening'`の場合、受け取ったモデルディレクトリへ`base.flax_model`を付ける。そのため、間違った引数は`models/initial.flax_model/base.flax_model`として扱われる。

これは対戦結果や候補の強さの問題ではなく、基準モデルとの評価を進められないパスの不具合である。

## 最小限の修正方針

`verified_opening`では`models/initial/`、従来の単体モデルbackendでは`models/initial.flax_model`を使う。全backendを無条件にディレクトリへ変えない。

例：

```python
reference_name = 'initial' if cfg.get('backend') == 'verified_opening' else 'initial.flax_model'
reference_checkpoint = Path(cfg['root']) / 'models' / reference_name
command[command.index('--opponent') + 1] = str(reference_checkpoint)
```

呼び出し側では、`verified_opening`に`base.flax_model`そのものを渡さない。入口でさらにファイル名が付くため、ディレクトリを渡す。

欠けている旧形式のファイルを新たに作って回避したり、参照評価を飛ばしたり、相手をランダム方策へ置き換えたりしない。判定基準・試合数・モデルの重みを変更しない。

## 必要な回帰テスト

一時rootとモックした評価ジョブで、実際の`evaluate_candidate()`を通す。実際の学習や128戦等は起動しない。

1. `verified_opening`のrootに初期base＋headを配置し、展開検査、候補対best、初期参照、確認評価まで進む条件を作る。初期参照で実在するディレクトリが渡され、2ファイルの読み込み対象が正しいことを確認する。
2. 旧単体モデルbackendでは、従来どおり実在する`initial.flax_model`を参照することを確認する。
3. 通常の評価条件・確認評価の別seed・採用条件を迂回していないことを確認する。

修正前に少なくとも2モデル形式のテストが失敗し、修正後に通ることを確認する。既存の関連テストも実行する。文字列が変わっただけを見るテストにしない。

## 保持すべき停止状態と候補

現在root：

```text
/home/hatk8/projects/yugioh-ai/experiments/continuous-elfnotes-openings-d16384-rehearsal-ready-20261109
```

確認時：`STOP=now`、`status=stopped`、pid/child_pidなし。

- completed_cycles：4
- cycle：5
- best：`models/initial`（採用モデルのまま）
- latest：`cycles/000005-715e84c3/candidate`
- pending.phase：`evaluating`
- pending.steps：4,096
- pending.training_seed：10000005
- pending.eval_seed：80004985

5回目の候補はすでに学習済みで、評価待ち。今回の修正で作り直したり削除したりせず、pendingの評価待ち状態を維持する。途中で止めた評価を完了した評価として流用しない。

修正前に、ソース・関連する評価ログ・state等の控えを残す。今回は監督側の参照先の修正なので、既存の学習snapshotを再生成して候補の由来を変えない。

ソースの版が変わったために、過去の実験ランナーが条件指紋の不一致で再利用を拒否する場合、それを回避するために旧記録のハッシュを書き換えない。

## 完了時の報告

差分、回帰テスト、修正前の控え、5回目候補と採用モデルのハッシュが不変であること、STOP・pendingが維持され学習／評価が起動していないことを報告する。

## 貼り付ける依頼文

```text
添付の不具合を、学習・対戦を停止したまま修正してください。

continuous_learning.pyの初期参照モデルの指定を、verified_openingではmodels/initialディレクトリ、旧単体モデルbackendでは従来のmodels/initial.flax_modelにしてください。evaluate_jobがディレクトリへbase.flax_modelを付ける前提を維持してください。

モックしたジョブでevaluate_candidateを通す回帰テストを追加し、候補対best→初期参照→条件を満たす場合の確認評価までのパスと、旧形式の維持を確認してください。実際の学習・対戦は起動しないでください。

5回目の候補、採用モデル、本番設定、snapshot、STOP、pending.phase=evaluatingを維持してください。再学習・候補採用・再開・評価の迂回は行わず、差分とテスト結果を報告してください。
```
