# Windows側起動スクリプトの復元（実行・起動なし）

2026年11月9日 19:11（日本時間）。

**復元は完了した。** スクリプトは元のバイト列のまま `C:\Users\hatk8\AppData\Local\YugiohAI\continuous-elfnotes.ps1` に戻した。現在のD条件の本番設定を参照していることも確認した。

次のことは行っていない。
- スクリプトの実行
- `Start`・`Install`・`DisableAutostart` などの操作
- Startupショートカットとタスクスケジューラへの変更
- STOPの削除
- 学習

## 1. 欠けていた原因

11月8日の `Install` は、Codexデスクトップアプリ（MSIXパッケージ `OpenAI.Codex_2p2nqsd0c76g0`）から実行された。そのため、MSIXのファイル仮想化により、`%LOCALAPPDATA%\YugiohAI` への書き込みがアプリ専用領域へ振り替えられていた。

- 振り替え先：`C:\Users\hatk8\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\YugiohAI\`
- 実際のパスには、フォルダ自体が存在しなかった。
- 無効化済みのタスクの前回結果は `0x8007010B`（ディレクトリ名が無効）で、この状況と一致する。

## 2. 復元したファイル

| ファイル | sha256 | 大きさ |
| --- | --- | --- |
| `continuous-elfnotes.ps1` | `ba92a90a82b5d6b9a2f981159a232b597ae038af0805c76301cdd840e63d96ed` | 4,806 |
| `autostart-mode.txt`（`startup`） | `b8af88be881954063dccf692f8e9b45816612c126fe0589542ecf1b4879a64c9` | 9 |

- スクリプトは、仮想化領域の原本と、Codex作業フォルダのソース（`Documents\Codex\2026-10-27\new-chat\work\continuous-elfnotes\`）と、バイト単位で一致した。内容は変更していない。
- `autostart-mode.txt` は、`DisableAutostart` がショートカットを消すために必要なので戻した。
- 古いログ（`host.log`・`host-error.log`）は戻さず、記録としてのみ保存した。
- 仮想化領域の原本は、削除していない。

## 3. 参照先の確認

| 確認 | 結果 |
| --- | --- |
| Windows側からの見え方（PowerShellで読み取りのみ） | 実際のパスに存在し、ハッシュ一致。構文解析エラーは0件 |
| `$LinuxConfig` | `/home/hatk8/projects/yugioh-ai/config/continuous-elfnotes.json`（本番設定、`7ba275dc…`） |
| `$LinuxScript` / `$Python` / WSL | `scripts/continuous_learning.py` / `/opt/ygo-ai/.venv/bin/python` / `Ubuntu-22.04`・`hatk8`（現環境と一致） |
| 本番設定の中身 | root `continuous-elfnotes-openings-d16384-rehearsal-ready-20261109`、学習率1.5e-5、復習 train・λ=5、bootstrap/cycle 4,096 |
| Statusが読む状態（同じ `status` コマンドをWSLで直接実行） | `worker_running` false、`stopped`・`idle`、pidなし、`training_condition` D、追加判断0 |

スクリプトは、rootではなく本番設定のパスを参照している。そのため、設定を切り替えるとスクリプトの参照先も追従する。

## 4. 各操作の動作（静的に読んだ結果。実行していない）

- **`Run`（Startupショートカットが使う）**
  - `continuous_learning.py run --config <本番設定>` を呼ぶ。
  - STOPがあると、`run_loop()` はハッシュ照合の後に `stopped` を記録して終了する。学習の前に止まり、STOPは消さない。
- **`Start`**：`resume` でSTOPを削除してから `Run` を起動する。**実行していない。**
- `Stop`：`stop --now` を実行する。
- `Status`：読み取りのみ。
- `DisableAutostart`：Startupショートカットを削除する。

## 5. 起動経路の現状

| 経路 | 状態 |
| --- | --- |
| Startupショートカット（`…\Startup\YugiohAI-Continuous-Elfnotes.lnk`、`82ee12e4…`） | 存在する。変更なし。**スクリプトの復元により、次回のWindowsログオンで `-Action Run` が実際に動く** |
| タスクスケジューラ `YugiohAI-Continuous-Elfnotes` | 無効。変更なし |

次回のログオン時に起きること：
- WSLが起動し、本番rootの `STOP=now` により学習せずに終了する。
- state.json が `stopped` で書き直され、`host.log` が新しく作られる。

ログオン時の起動そのものを止めたい場合は、`-Action DisableAutostart` でショートカットを外せる。ただし、今回は指示がないので実行していない。

## 6. 停止状態（作業前後で同じ）

- 4つのroot（D用の現本番、d16384、v2、v1）はすべて `STOP=now`、`stopped`、pid・pending・child_pidなし。
- 学習のプロセスはない。
- 本番設定のハッシュは変わっていない。

## 7. 記録

`experiments/windows-controller-restore-20261109/`：

- `restore-record.json`（`d6caf5e8…`）
- `status-after.json`
- `code/restore_controller.py`（`b9a5e628…`）。前提の照合、上書き拒否、ハッシュ照合を行う。
- `source/`：仮想化領域の4ファイルとショートカットの写し

## 8. 元に戻す

`C:\Users\hatk8\AppData\Local\YugiohAI\` を削除すると、復元前の状態に戻る。復元前は、スクリプトが無いのでログオン時の起動が失敗していた。
