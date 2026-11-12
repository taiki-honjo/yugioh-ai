# Claude向け：アルトメギアAIの研究・学習を進めるコンテクスト

更新日：2026年11月10日（日本時間）

## 1. ユーザーの最新の依頼と優先順位

ユーザーの依頼：

> B2Bは後回しでいい。あとでClaudeにお願いする。アルトメギアのをClaude用に作成して。

これからの主作業は、**アルトメギア自身を扱うAIの研究・学習・評価**。将来、エルフェンノーツ＋キラーチューンとの異種デッキ対戦にも使えるようにする。

- B2Bの学習seed追試、追加学習、追加対戦は今回始めない。既存の追試仕様書は保留として残す。
- エルフェンノーツの本番、採用モデル、停止中の5回目の候補、STOP、pendingには触れない。通常運転を再開しない。
- この文書の作成時点で、アルトメギアの学習や実エンジン評価は新たに実行していない。
- 以下の実験回数・着手順はCodexからの具体的な初回作業案。ユーザーが指定した予算と混同しない。

## 2. 環境

| 項目 | 値 |
| --- | --- |
| WSL | Ubuntu-22.04、ユーザー `hatk8` |
| プロジェクト | `/home/hatk8/projects/yugioh-ai` |
| Python | `/opt/ygo-ai/.venv/bin/python`。最終symlinkをresolveした別Pythonを使わない |
| Windowsからの参照 | `\\wsl.localhost\Ubuntu-22.04\home\hatk8\projects\yugioh-ai` |
| 現行ランタイム | `/home/hatk8/projects/yugioh-ai/evaluation/opponent-article-basics-20261108/runtime` |
| native SHA256 | `19a60f38a3d39b61a2ee082168d5a788a1022b2eb569633816596c8b3de21294` |
| 語彙 | `/home/hatk8/projects/yugioh-ai/experiments/meta-opponents-20261107/run-20261107-224631/code_list.txt` |
| 語彙SHA256 | `aeb5be89d4a5591ccc5cce6af96def74ef47463a55265f75c11b92029e772e7b` |

Windowsからの実行例：

```powershell
wsl -d Ubuntu-22.04 -u hatk8 --exec /opt/ygo-ai/.venv/bin/python -B /home/hatk8/projects/yugioh-ai/<script.py>
```

`<script.py>`は説明用の置き換え箇所であり、そのまま実行するコマンドではない。スクリプトごとのcwd・PYTHONPATH・環境変数も確認する。

## 3. すでに準備したアルトメギア専用領域

```text
/home/hatk8/projects/yugioh-ai/experiments/artmage-parallel-20261110/
  STOP                         # now。最初に作成したファイル
  inventory.json               # 現在の資産、パス、ハッシュ、確認範囲
  plan.json                    # 初回の有限確認案（未実行）
  source-integrity.json        # 元ファイル20件の前後ハッシュ、変更0
  models/initial/
    base.flax_model            # 現在のArtmageモデルの固定コピー
    chain-head.flax_model
  decks/Artmage.ydk            # 現在のレシピの固定コピー
  snapshot/
    registry-source.json
    article-manifest.json
    article-delivery.json
```

モデル2ファイルとデッキのコピーは、原本とのハッシュ一致・書き込み禁止を確認済み。状態は **prepared_stopped**。学習監督プロセス・自動起動・実行可能なArtmage専用の常時学習設定を設置したわけではない。

新しい実行結果・コードはこの領域、またはその下の新しいrunディレクトリへ保存する。旧runや共通レジストリの書き換えで実験対象を切り替えない。

## 4. 現在の開始モデルとデッキ

現行登録は次のファイル：

`/home/hatk8/projects/yugioh-ai/config/latest-deck-position-policies.json`

Artmageは**base＋chain-headの2ファイル形式**。2026年11月7日の単体モデルのままではない。

| 項目 | 値 |
| --- | --- |
| base SHA256 | `12500c7724b2d723983a2963a960d75172a6df51fb55ea4f3acf102eeb34132e` |
| chain-head SHA256 | `0bc7534cab102cd87586e5d22bd4ad0cd3b55d52d08e284146976506ae6193fc` |
| デッキ | メイン40枚、EX15枚、サイド0枚 |
| デッキSHA256 | `a93c4d30d0945f58f741f1e135fdcbc0b0c438aff575cf6d0d02ce9a82c381d6` |
| 表示形式補助SHA256 | `4fe74c83a460ee3565c3ff276ce2d080b233a6152439ec92147ba90d60d46f90` |

原本：

```text
/home/hatk8/projects/yugioh-ai/experiments/opponent-article-training-20261108/run-20261108-145203/training/Artmage/step-0200/base.flax_model
/home/hatk8/projects/yugioh-ai/experiments/opponent-article-training-20261108/run-20261108-145203/training/Artmage/step-0200/chain-head.flax_model
/home/hatk8/projects/yugioh-ai/experiments/meta-opponents-20261107/run-20261107-224631/decks/Artmage.ydk
```

新領域内の固定コピーを、今後の実験の開始点として使う。`step-0200`は旧記事展開の教師あり学習で選択されたcheckpoint。200判断のPPO学習という意味ではない。

履歴として、もっと前に69,632判断の単体モデルがあり、そこから記事の基本展開を追加学習した。古い `multideck_opponent_inventory_2026-11-09.json` は単体モデルを記載しているため、現在の開始点として優先しない。

## 5. 現行の推論入口と補助

入口：

`/home/hatk8/projects/yugioh-ai/scripts/deck_policy.py`

```python
from scripts.deck_policy import create_policy
policy = create_policy('Artmage')
policy.reset()
action = policy.act(obs, info)
```

これは現在の登録済みArtmageを読む入口の説明である。新しい候補の固定コピーを読むには、隔離した登録情報を指定できる実装が必要。現在の関数は共通のレジストリを直接読む。共通ファイルを候補用に上書きして回避しない。

推論コード：

`/home/hatk8/projects/yugioh-ai/experiments/opponent-article-training-20261108/run-20261108-145203/code/opponent_article_bc_common.py`

SHA256：`f4c3903e3a4841b97f6ffb8840bb4cfa5513b5abfead9ff41fa7c785ceba6cce`

表示形式補助：

```text
/home/hatk8/projects/yugioh-ai/evaluation/all-deck-position-20261109/code/battle_position.py
/home/hatk8/projects/yugioh-ai/evaluation/all-deck-position-20261109/code/position_cards.json
```

この補助は、公開情報を用いて攻撃・守備表示を選ぶ共通補助。戦闘全般の最適な行動を保証するものではない。Artmageにエルフェンノーツ専用の中央配置補助を付けない。

学習や評価では、対戦ごとのRNN・補助の記憶のリセットと、両プレイヤーの状態分離を維持する。相手の非公開手札・デッキ順・採点用の目標盤面をモデルに渡さない。

## 6. 既存の学習データと確認結果

データ領域：

`/home/hatk8/projects/yugioh-ai/experiments/opponent-article-training-20261108/run-20261108-145203/dataset/Artmage/`

| 区分 | 系列 | 判断 | SHA256 |
| --- | --- | --- | --- |
| train.npz | 12 | 1,077 | `61bc7cb52209c4c15b97e952694dd4a748752f59ea1734f99f4dd5ed197f3a14` |
| validation.npz | 2 | 158 | `1022ba94b41d0d7cbe6d5d993eeb3a8edabb5d00b2e03ac2ae4b5ae47ddd04e2` |
| holdout.npz | 4 | 364 | `00c320a8f611df5f35dcbfe1f423bb3eadf7302a5cef3b738b98cd1128e1e504` |

12本は「襲撃」の基本展開を複数初手・順序で学ぶ系列であり、12種類の初動や妨害対応を学習済みという意味ではない。

旧manifest：

`/home/hatk8/projects/yugioh-ai/experiments/opponent-article-training-20261108/run-20261108-145203/manifest.json`

関連コードは同runの `code/` にある：

- `article_witness.py`
- `prepare_opponent_article_training.py`
- `train_opponent_article_bc.py`
- `evaluate_opponent_article_bc.py`
- `battle_opponent_article_bc.py`

旧記事学習の設定は、600更新、batch size 2、base学習率3e-5、head学習率3e-4、KL 0.01。validationの教師強制CEでstep-0200を選択した。**この設定は履歴であって、新しいPPO学習へ無条件に転用する推奨値ではない。**

過去に記録された自由実行の確認初手は `Artmage-g08-p0`〜`Artmage-g11-p0` の4件。表示形式補助を付けた状態でも **4/4成功**。

記録：

`/home/hatk8/projects/yugioh-ai/evaluation/all-deck-position-20261109/verified/regression/Artmage/after/`

この4件は元はholdoutだったが、すでに結果を何度も見ている。今後は既知の回帰確認として扱い、未使用の最終試験とは呼ばない。今回のコンテクスト作成では再実行していない。

「アルトメギアの全初手・妨害対応・対戦能力が完成している」とは解釈しない。旧対戦結果も、今の相手・重み・ルールでの勝率として流用しない。

## 7. 展開の参考資料と未確認の範囲

プロジェクトの参照一覧：

`/home/hatk8/projects/yugioh-ai/docs/expansion-references-2026-11-08.md`

記事：[アルトメギアデッキ紹介／鈴木](https://note.com/kind_borage903/n/n0b8910935f9a)

| 優先度 | 条件 | 参照 |
| --- | --- | --- |
| 既存 | 襲撃初動 | https://duel-simulator.miyauchidp.dev/replay/g2rN57Xb |
| 初回の追加候補1 | メディウス＋モンスターコスト | https://duel-simulator.miyauchidp.dev/replay/wsMMwXxE |
| 初回の追加候補2 | 学都＋魔法罠コスト1枚 | https://duel-simulator.miyauchidp.dev/replay/XUS4azXa |
| その後 | メディウス＋魔法罠コスト、獄神獣、学都＋コスト2枚、テルミナスなど | プロジェクトの参照一覧 |

記事と、選択した2本のURLは調査した。記事本文の内容は読めたが、この調査のweb取得ではリプレイの中身は取得できなかった。リンクを掲載しただけで、全操作を見た・エンジンで再現できた・現レシピで合法だと認定しない。

初手、コストの種類と残量、途中の分岐、最終盤面、相手ターンの用途を、実際の参照内容とエンジンで確かめる。本文の想定をそのまま正解ラベルにしない。カードテキスト・裁定を確認する場合は公式カードDBを優先する。

## 8. 初回にClaudeへ進めてほしい作業

### A. 参照・起動経路の整理（実エンジンを使わない準備）

1. この文書と `inventory.json` を読み、開始モデル・デッキ・語彙・コードのハッシュを現物と照合する。
2. MDの現在のカードプールと禁止・制限の全体を、現在の40/15レシピと照合する。MDとOCGを混同しない。
3. Artmage用の隔離したモデル・デッキ・評価入口を実装する。共通レジストリやElfnotesの設定を切り替えない。
4. 既存のsplitを維持し、新しい手本候補の初手が旧validation・holdoutと重ならないことを検査する。同じ完全初手の順列は同じグループにまとめる。
5. コマンド・データ・コード・開始モデルのハッシュとseed、次の8回上限を実行前に固定する。

MD制限の参考：[2026年11月5日適用の改訂一覧](https://game8.jp/yugioh-masterduel/423063)。一覧の最新変更だけを見て、全レシピの合法性を検査済みとしない。今回の準備では全リスト照合は未完了。

### B. 最初の有限確認案：実エンジン上限8回

| 内容 | 回数 |
| --- | --- |
| 登録済みArtmageモデル＋同じ表示形式補助で、既知4初手の回帰確認 | 4 |
| メディウス＋モンスターコストの手本の記録と再現 | 2 |
| 学都＋魔法罠コスト1枚の手本の記録と再現 | 2 |
| 合計 | **8** |

学習・通常の対戦・採用・サーバーへの新しいリプレイ投稿は、この8回には含めず、初回確認では行わない。失敗した実行も回数に含める。旧台帳は使わず、台帳を巻き戻して回数を増やさない。

新初動の初手は、学習や結果閲覧の前に固定する。途中で合法な選択肢から予定の行動が消えた場合は、原因とその時点の合法な選択肢を保存する。失敗を「そのカードでは初動になれない」と一般化せず、また都合のよい初手を探す追加実行で予算を超えない。

新しい手本は、観測・全操作の意味・開始seed・履歴・終了情報・各カードの場所を保存する。自分のターン終了時だけでなく、相手ターンに使う能力と次の自分ターンの後続も、確認済みと想定を分けて記録する。

デュエルが終了した段では、env.stepの戻り値で終局を処理し、破棄済みデュエルに `benchmark_snapshot()` 等を呼ばない。以前のB2B手本で、その呼び出しが原因と推定されるsegfaultがあった。

### C. 追加学習へつなげる成果物

初回確認の成果として、次を揃える。

- 既存4件の成功集合と、原本の記録との対応。
- 新初動2条件の合法な手本、モデルへ渡した観測、再現結果、データ化できる判断数。
- 元のtrainと、新しく追加できるtrain用手本のmanifest。原本のNPZや旧合否を上書きしない。
- Artmage用の保持試験と、新しい条件を確認する試験の仕様。
- 最初の有限学習の計画。目安は候補1つ・4,096環境判断だが、互換性・損失・復習データ・評価予算をArtmage用に決めて登録する。

PPOと復習を組み合わせる場合、現行のElfnotes用コードのDECK固定、専用のケース・配置・ラベル・データ検査を、そのままArtmageへ流用しない。Artmageのbase＋headを両方正しく更新・保存・読込できることを確認する。

## 9. 評価設計で守りたいユーザーの意図

対戦の強さを伸ばすのが目的。最終盤面が一種類に完全一致することだけで、新しい展開を落とさない。

- 複数の合法で有効な着地を認め、到達までの消費と、残る妨害・手札・後続を記録する。
- 盤面に能力が存在するかと、モデルが実際に使うかを分ける。
- 自分ターン終了、相手ターン終了、次の自分ターン終了を分ける。
- ディアクトロスの無効、覚醒と融合の組合せ、リテラの資源回復などを評価候補にできるが、カード文・合法性・実際の解決を確認するまでは未確認とする。
- Gやドロールを、直接無効にした回数が0だから低価値としない。ドロー・手札追加制限・相手の展開の変化・後続を別に測る。両者を単純に加算しない。
- 発動できる効果を使わなかったこと、主力を別の展開の素材にしたことだけで悪手と断定しない。
- 対戦は、まずArtmageの新旧モデルのミラー比較ができる構成を用意する。その後、固定したElfnotes等との異種デッキ対戦へ広げる。各相手別の成績を残す。
- 先後ペア、未確認・異常・打ち切り・timeout・疑いの区別は、既存のレビュー済み評価仕様を維持する。

## 10. B2B研究から得た注意点（今回B2B作業を再開する指示ではない）

- 同じ初手のデモが少数でも、共有重みを通じて別の行動が変わる。特定区間の損失を0にしても、そこが不変になる保証はない。
- 文脈の有効マスクと、教師損失の対象マスクを別にする。文脈をpaddingとして削らずRNNへ通す。
- 系列の抽出割合は、損失や勾配への寄与の割合ではない。出典ごとの抽出数・損失対象数・CE・勾配を残す。
- クリップ前の勾配ノルムだけで、Adam後の更新方向や改善原因を断定しない。
- 既知の展開を保持しても、対戦での強さが上がるとは限らない。行動・保持・勝敗を別々に報告する。
- 保存した操作列の再現と、別の学習seedで改善が再現することは別の確認。
- 実際に未使用だった試験も、結果を見た後は未使用の最終試験と呼ばない。

## 11. 実行資源と停止状態

- アルトメギアの研究領域は独立しているが、GPUの学習・推論ジョブは同時に1つまでを初期方針とする。これは準備ファイルに記載した方針であり、共有ロックを実装済みという意味ではない。
- 既存ジョブがあれば、今回の作業を理由に強制終了しない。空きを確認して実行する。
- 現行の資源上限の参考は、ジョブメモリ5GiB、swapなし、CPU上限300%、ジョブ上限1,800秒。Artmageの実行前に設定を固定する。
- 新領域のSTOPは準備段階では `now`。有限ジョブに停止制御を接続する場合、実験側だけを対象とし、終了・失敗・中断でもSTOP復元と子プロセス終了を行う。
- WindowsのStartup、タスク、コントローラ、ElfnotesのSTOPは変更しない。ログオンによる既存監督の時刻更新と、研究作業による変更は区別する。

## 12. 次のClaudeの報告に含めること

1. 何を実装・実行したか、どの上限で何回使ったか。
2. MDの制限とデッキの照合結果、採用した固定デッキの版。
3. 既存4件の回帰結果、新初動の成立/不成立/未確認を分けた表。
4. 参照記事の説明、台本による証拠、モデル自身の自由実行を区別した結果。
5. モデル・データ・コード・デッキ・評価条件のハッシュと再実行手順。
6. 学習に進むためのArtmage固有の変更点、初回の学習・評価計画。
7. B2B保留、Elfnotes本番・候補・旧資産が保たれていることと、残存プロセスの確認。

研究上の通常の実装判断は、依頼の範囲で進めてよい。未確認な情報を確認済みに変えたり、予算・採用条件・本番運転を結果に合わせて後から変えたりしない。
