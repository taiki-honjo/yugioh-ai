# 大学PCへの引き継ぎ（2026-11-12）

このリポジトリは、自宅の研究用プロジェクトから作成した非公開のソース・データの移行用スナップショットです。
clone・検証だけでは学習、対戦、自動起動、候補採用を実行しません。

## 取得とファイル確認

大学PCで、そのPC専用のSSH鍵をGitHubへ登録してから、LinuxまたはWSLで実行します。

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:taiki-honjo/yugioh-ai.git
cd yugioh-ai
python3 migration/verify_transfer.py
```

verify_transfer.py は標準ライブラリだけで、同梱したモデル・データ・ソースのハッシュとリンクを確認します。
エンジンやモデルを読み込まず、学習も対戦もしません。リンク先が絶対パスのものは別に一覧表示します。

## 含めたもの

- scripts・tests・docs・設定と、experiments/evaluation内の研究コード、C++/Luaのソース、各種manifest・登録計画・集計。
- 手本のnpzと、Artmageの40系列を読み込むために必要な元のrecord.json。
- Artmageの初期モデル、手札分岐の研究モデル、32,768判断後の研究候補（base＋chain-head）。
- Elfnotesの採用モデルと停止中の5回目候補、現在の本番rootの状態とSTOP。
- 2026年版エンジンのソース、カードスクリプト、DB、現在使っているPython 3.10/Linux x86_64用のネイティブライブラリ。
- ソースと同梱物の出所を記した既存のenv manifest、第三者のライセンス。

過去の全モデル、詳細な対局軌跡の大半、ビルド生成物、仮想環境そのものは含めていません。
元の記録は自宅PCに残しています。過去の報告書にあるすべてのパスが同梱されているわけではありません。
Git履歴にない過去の重み・対局軌跡が必要な追試では、必要な資産を別途持ち出します。
全体の移行対象は tracked-files.jsonl、直近の続きに必要な固定ファイルは critical-files.json にあります。

## 現時点の研究状態

Artmageの直近候補:

- base: 7f4ee702c09ee2d2bdf00969c7afe701a7a050b71422662802625de607c28413
- head: 4f099640b98c3fd6b100889c6d4e127a01853b130b6c961a67c2c1519330599a
- 保存場所: experiments/artmage-rl-training-20261112/train/candidate/
- 学習32,768環境判断、評価128戦の報告は docs/artmage-rl-training-2026-11-12-report.md。
- 終了処理の検査でエラーになった学習から、例外を記録して保存済み重みを取り出した研究候補です。
  正式な採用モデルではなく、元の「不成立」記録も保持しています。
- 引き分けの扱いと後片付けの検査には未修正の問題があります。
  docs/artmage-rl-scaleup-plan-2026-11-12.md は次段階の指示書で、実装済みとは扱いません。
- 3ファイル形式の応答adapterは今回のRL学習に混ぜていません。

比較相手・復習元の基準モデルは experiments/artmage-hand-branch-20261110/train/candidate/ の
base b9c7751f… / head 77191d5d… です。重みの再開は最適化状態まで含む完全な再開ではありません。

Elfnotesの本番はD条件、STOP=now、stopped、5回目候補がpending=evaluatingのままです。
移行作業で学習・対戦・採用を行っていません。B2Bは保留です。

## 実行環境の設定は大学PCで必要

自宅はUbuntu 22.04 / Python 3.10.12 / NVIDIA GPUで、Python環境は /opt/ygo-ai/.venv です。
インストール済みパッケージの一覧は requirements-home-20261112.txt に保存しました。
CUDA関連も含む自宅の一覧なので、大学PCのOS・GPU・ドライバーを確認せず一括適用しないでください。

現在の研究コード・manifest・一部のシンボリックリンクには
`/home/hatk8/projects/yugioh-ai` と `/opt/ygo-ai/.venv` という絶対パスが残っています。
cloneだけでは、別ユーザー名や別ディレクトリからそのまま実行できません。
大学PCでの設定時に、互換パスを用意するか、新しい実験領域でパスと指紋を登録し直します。
古い固定manifestを一括置換すると記録されたハッシュも変わるので、過去の証拠を書き換えないでください。

同梱のnativeはLinux x86_64 / CPython 3.10用です。Windowsネイティブや異なるPython ABIでは使えません。
リンク先には libsqlite3、Lua 5.4、glog、gflags、libunwind などが必要です。
環境が違う場合は、同梱のCMakeLists.txt・env manifestを基に別の実験runtimeで再ビルドします。
Artmage用CMakeには、Elfnotes runtimeのbuild内の静的ライブラリを参照する箇所があります。
build生成物は移していないので、ソースからの再ビルドでは先にその依存関係を作る必要があります。

従来の青眼用 /opt/ygo-ai/current は外部環境のため含めていません。
WindowsのStartupショートカットやタスク、SSH秘密鍵、GitHub認証情報も移しません。

## 大学側のClaude/Codexへ渡す内容

1. このファイルとREADME、docs/artmage-rl-training-2026-11-12-report.md、
   docs/artmage-rl-scaleup-plan-2026-11-12.mdを読む。
2. 大学PCのOS・CPU・GPU・ドライバーとPythonを確認し、移行したファイルをverify_transfer.pyで照合する。
3. 元の研究資産を変えない新しい実験領域で、パスと実行環境を準備する。
4. 学習を再開する前に、引き分け処理と終了検査の未修正事項を扱う。
5. 自宅の本番・STOP・pendingと過去の台帳は維持する。学習の開始・量・対戦は別の作業として扱う。
