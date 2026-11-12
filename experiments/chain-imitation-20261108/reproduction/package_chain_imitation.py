"""Export the completed experiment without promoting its weights to production."""
import csv,difflib,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
P=Path('/home/hatk8/projects/yugioh-ai')
B=P/'experiments/chain-imitation-20261108'
W=Path('/mnt/c/Users/hatk8/Documents/Codex/2026-10-27/new-chat/work')
O=W.parent/'outputs/chain-imitation-2026-11-08'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def copy(s,d):
    d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d)
assert read(B/'validation-complete.json')['status']=='complete'
tr=read(B/'training/result.json');ev=read(B/'evaluation/summary.json');rt=read(B/'runtime.json')
assert sha(tr['source_model'])=='10be73b736ce4751b118450cfdff3c52b2183d6e84baae3af3c518fa40f843bb'
assert sha(P/'scripts/evaluate.py')=='67db15ffeb9f94884dda433352f52170ac0b7d60de46056c88f0317177d9d7d0'
assert sha(B/'scoring-evaluator.py')==sha(P/'scripts/evaluate.py')
assert read(B/'normal-mode-regression.json')['passed']
assert ev['results']['baseline']['diagnostic24']['passed']==3
assert sha(B/'dataset/split-manifest.json')==tr['dataset_manifest_sha256']
assert sha(rt['native'])==rt['sha256']
assert not tr['holdout_used_for_training_or_selection']
fixed=P/'experiments/combo-benchmark-20261107/evaluation-v1/runner.py'
assert sha(fixed)=='74b9baf65b8e46593272e5b9676727fc7fb85d20ea1e24051b4a5f88dea5333e'
assert (B/'combo.py').read_text()==fixed.read_text().replace("B=P/'experiments/combo-benchmark-20261107'","B=P/'experiments/chain-imitation-20261108'")
modern=next((P/'experiments/elfnotes-20261106/runtime/agent/ygoenv/ygoenv/ygopro').glob('*.so'))
assert sha(modern)=='57aa077f335c5c3e3cee7a91a58b03806ce4b6f439cbbbc7179db650b1867f2c'
O.mkdir(parents=True,exist_ok=True)
for folder in ['dataset','evaluation','battle','counter-probe','normal-mode-regression']:
    for s in sorted((B/folder).rglob('*')):
        if s.is_file() and 'init-error' not in s.parts:copy(s,O/s.relative_to(B))
for name in ['runtime.json','self-ash-audit.json','normal-mode-regression.json','validation-complete.json','scoring-evaluator.py']:
    copy(B/name,O/name)
for s in B.glob('memory-*.json'):copy(s,O/'resource-records'/s.name)
for s in B.glob('*.log'):copy(s,O/'logs'/s.name)
for name in ['result.json','history.json','gpu.csv','best-checkpoint.txt']:
    copy(B/'training'/name,O/'training'/name)
copy(B/'training.log',O/'training/training.log')
selected=Path(tr['selected_checkpoint'])
for name in ['base.flax_model','chain-head.flax_model']:copy(selected/name,O/'model'/name)
copy(tr['source_model'],O/'model/baseline.flax_model')
for s in sorted(W.glob('*chain*.py')):copy(s,O/'reproduction'/s.name)
for name in ['audit_self_ash.py','run_ygo_memory_limited.py','combo_fixture_hooks.h','check_combo_runtime_regression.py','eval_meta_opponents.py']:
    copy(W/name,O/'reproduction'/name)
copy(B/'combo.py',O/'reproduction/frozen-combo-runner.py')
copy(B/'check-normal-regression-used.py',O/'reproduction/check-normal-regression-used.py')
source_hashes={p.name:sha(p) for p in sorted((O/'reproduction').glob('*')) if p.is_file() and p.name!='source-hashes.json'}
write(O/'reproduction/source-hashes.json',source_hashes)
for p in (O/'reproduction').glob('*'):
    if p.is_file():copy(p,B/'reproduction'/p.name)
old=P/'experiments/combo-benchmark-20261107/runtime/agent/ygoenv/ygoenv/ygopro/ygopro.h'
new=B/'runtime/agent/ygoenv/ygoenv/ygopro/ygopro.h'
copy(new,O/'runtime-source/ygopro.h')
copy(B/'runtime/CMakeLists.txt',O/'runtime-source/CMakeLists.txt')
diff=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='benchmark-runtime/ygopro.h',tofile='chain-runtime/ygopro.h'))
(O/'runtime-source/public-chain.patch').write_text(diff)
copy(rt['native'],O/'runtime-source'/Path(rt['native']).name)
meta=Path((P/'experiments/meta-opponents-20261107/latest-run.txt').read_text().strip())
mf=read(meta/'manifest.json')
copy(mf['vocabulary'],O/'vocabulary/battle-code-list.txt')
copy(P/'experiments/elfnotes-20261106/runtime/agent/scripts/code_list.txt',O/'vocabulary/combo-code-list.txt')
assert (O/'vocabulary/battle-code-list.txt').read_text().splitlines()[:37]==(O/'vocabulary/combo-code-list.txt').read_text().splitlines()
for s in (meta/'decks').glob('*.ydk'):copy(s,O/'decks'/s.name)
copy(P/'assets/deck/elfnotes-kewltune-decklist.json',O/'decks/elfnotes-kewltune-decklist.json')

summaries={}
for policy in ['baseline','bc_with_chain']:
    summaries[policy]={}
    for deck in ['Artmage','ClownTroupe','BrandedDracotail','SkyStriker']:
        s=B/'battle'/policy/deck/'summary.json';d=read(s);summaries[policy][deck]=d
        rows=list(csv.DictReader((s.parent/'episodes.csv').open()))
        raw=d['matchup']['episodes'];assert len(rows)==len(raw)==64
        assert all(all(row[k]==str(value) for k,value in record.items()) for row,record in zip(rows,raw))
        assert d['meta']['lua_errors']==0 and d['meta']['resources']['live_duels']==0
        assert d['meta']['raw_options_peak']<=128
        assert d['meta']['head_opponent_batches']==0
        assert d['meta']['evaluator_sha256']==sha(B/'scoring-evaluator.py')
        assert d['meta']['vocabulary_sha256']==sha(mf['vocabulary'])
        assert sha(d['meta']['opponent'])==d['meta']['opponent_sha256']
        assert sha(d['meta']['candidate'])==d['meta']['candidate_sha256']

# Matched seed-pair differences are only a small, conditional regression screen.
paired={};rng=np.random.default_rng(8118)
for deck in summaries['baseline']:
    def groups(d):
        out={}
        for r in d['matchup']['episodes']:out.setdefault(r['pair_id'],[]).append(r)
        assert all(len(v)==2 for v in out.values())
        return out
    oldg=groups(summaries['baseline'][deck]);newg=groups(summaries['bc_with_chain'][deck])
    assert oldg.keys()==newg.keys()
    ids=[i for i in oldg if not any(r['abnormal'] for r in oldg[i]+newg[i])]
    delta=np.array([sum(r['win']+.5*r['draw'] for r in newg[i])/2-sum(r['win']+.5*r['draw'] for r in oldg[i])/2 for i in ids])
    bootstrap=rng.choice(delta,(10000,len(delta)),replace=True).mean(1) if len(delta) else np.array([])
    paired[deck]=dict(common_complete_pairs=len(ids),excluded_pairs=[i for i in oldg if i not in ids],
        score_difference=float(delta.mean()) if len(delta) else None,
        paired_bootstrap_ci95=np.quantile(bootstrap,[.025,.975]).tolist() if len(delta) else None,
        caveat='Exploratory, common naturally completed pairs only; 32 pairs maximum, four comparisons, censoring may bias results.')
write(O/'battle/paired-comparison.json',paired)

names={0:'—',13597785:'ルキナ',59581480:'ディーナ',85976588:'フォルトナ',56651978:'レギナ',
       64491754:'パラレル',24092792:'ラプソディア',43904702:'クリップ',16509007:'ミクス',42781164:'トラックメイカー',
       14558127:'灰流うらら',97268402:'エフェクト・ヴェーラー',94145021:'ドロール＆ロックバード',23434538:'増殖するG'}
acts={'SpSummon':'特殊召喚','Summon':'通常召喚','MSet':'裏側守備表示でセット','Activate':'効果発動','Cancel':'見送り','Set':'セット','None':'選択','Repo':'表示変更'}
for label in ev['results']:
    for split in ['diagnostic24','holdout','chain_holdout']:
        for p in (O/'evaluation'/label/split).glob('*.json'):
            d=read(p);lines=[f"# {d.get('id',p.stem)}",'',f"成功: {d['passed']}",'']
            for row in d['trace']:
                a=row['selected'];code=a['code'];card=names.get(code,str(code))
                detail=f" / 場所{a['place']}" if a.get('place') else ''
                if a.get('phase') not in [None,'None']:detail+=' / '+a['phase']
                chain=' → '.join(f"P{x['player']}:{names.get(x['code'],str(x['code']))}" for x in row.get('public_chain',[])) or 'なし'
                lines.append(f"- {row['step']}: P{row['player']} {acts.get(a['act'],a['act'])} {card}{detail}（処理待ち: {chain}）")
            if 'critical' in d:
                lines+=['',f"所定の発動判断: {d['action_correct']} / 効果が処理された: {d['search_or_recruit_resolved']}"]
            p.with_suffix('.md').write_text('\n'.join(lines)+'\n')

family_ja={'rukina_search':'ルキナのサーチ','dina_place':'ディーナの配置','fortuna_place':'フォルトナの配置','regina_recruit':'レギナの展開',
           'dina_chain':'ディーナを含む連続展開','rukina_chain':'ルキナを含む連続展開','clip_hand_synchro':'場クリップ＋手札ミクス','mix_hand_synchro':'場ミクス＋手札クリップ'}
table=['| テスト | 元モデル | 学習後＋公開チェーン機能 | 同じ学習後・追加機能を外す |','|---|---:|---:|---:|']
for title,key in [('従来の固定ケース（学習未使用）','diagnostic24'),('別の初手の展開','holdout'),('自分／相手への発動判断','chain_holdout')]:
    values=[f"{ev['results'][p][key]['passed']}/{ev['results'][p][key]['total']}" for p in ev['results']]
    table.append('| '+title+' | '+' | '.join(values)+' |')
battles=['| 相手 | 元モデル（勝/敗/分） | 学習後（勝/敗/分） | 自然終了した共通ペアのスコア差 |','|---|---|---|---|']
deck_ja={'Artmage':'アルトメギア','ClownTroupe':'道化の一座','BrandedDracotail':'烙印ドラゴンテイル','SkyStriker':'閃刀姫'}
for deck in deck_ja:
    vals=[]
    for policy in summaries:
        d=summaries[policy][deck]['complete_natural_pairs']
        vals.append(f"{d['wins']}/{d['losses']}/{d['draws']}（{d['episodes']}戦、勝率{d['win_rate']:.1%}、95%区間{d['ci95'][0]:.1%}–{d['ci95'][1]:.1%}）")
    p=paired[deck];ci=p['paired_bootstrap_ci95']
    delta=f"{p['score_difference']:+.1%}（参考区間{ci[0]:+.1%}–{ci[1]:+.1%}、{p['common_complete_pairs']}ペア）" if ci else '算定不能'
    battles.append('| '+deck_ja[deck]+' | '+' | '.join(vals)+' | '+delta+' |')
abnormal=sum(sum(r['abnormal'] for r in d['matchup']['episodes']) for ds in summaries.values() for d in ds.values())
resource=[read(p) for p in (O/'resource-records').glob('*.json')]
peak=max(d['peak_bytes'] for d in resource if d['exit_code']==0)/2**20
contract=dict(status='experimental_not_promoted',base_sha256=sha(selected/'base.flax_model'),head_sha256=sha(selected/'chain-head.flax_model'),
    inference='base logits + learned ChainHead(features, actions_); argmax over the unchanged native legal action mask',
    runtime=rt,combo_vocabulary_sha256=sha(O/'vocabulary/combo-code-list.txt'),battle_vocabulary_sha256=sha(mf['vocabulary']),
    token_contract='415 battle entries preserve the first 37 combo token IDs, 1000 embedding rows',
    required_features=['public_chain_length','role relative to info.to_play: 0 none / 1 self / 2 opponent','public_chain_card_id','public_chain_effect'],
    source_checkpoint=tr['source_model'],source_sha256=tr['source_model_sha256'],requires_head=True,
    constraints='The exported base alone is an evaluated head-removal condition, not this candidate. New native info adapter is required. Do not load with the original headless CLI.',
    environment='Ubuntu-22.04 WSL; /opt/ygo-ai/.venv; fixed original modern dependencies; local existing paths required')
write(O/'model/runtime-contract.json',contract)

report=f'''# 展開の手本学習と誤った自己妨害の検証（2026-11-08）

短い展開手順の成功率と、今回用意した自己妨害の判断は改善しました。対戦成績は下表のとおりです。これは基本手順を覚えたかの検証であり、環境全体での最適な展開やカード効果の完全な理解を示すものではありません。

## 実測した結果

{chr(10).join(table)}

自分の効果を通す問題は0/10→10/10、相手の効果を灰流うららで止める問題は8/8→8/8でした。参照手順の正解ラベルだけでなく、実エンジンでサーチ・リクルートが処理された／止まったことを判定しています。

追加機能を外した条件もすべて成功しました。これは共同学習したモデルから推論時だけ機能を取り除いた比較です。追加情報なしで別途学習した比較ではありません。新しいチェーン情報が改善の原因・必須条件だったとは結論できません。

## 対戦への影響

相手AI4体を固定し、元モデル・学習後をそれぞれ64戦、合計512戦で比較しました。各相手32初期乱数×先後交代、学習・以前の対戦と異なるseed 11037です。各相手AIの学習量は4,096＋65,536判断で揃え、重みとデッキを固定しています。

{chr(10).join(battles)}

異常終了は合計{abnormal}戦。表は異常を含む初期乱数の先後ペア全体を除いた参考成績です。32ペアまでの小規模比較で、4相手を調べています。区間と差は探索的な参考値です。相手の学習量が等しくても相手AIの熟練度が等しいとは限りません。

利用者が更新した評価器をそのまま凍結し、元の出力はbattle/*/*/summary.jsonに保存しました。ネイティブの判断上限到達時の報酬0をLP判定済みの引き分けとは扱いません。上限到達等があれば、そのペアを丸ごと除いた自然終了部分を別欄にし、除外の偏りが残ることを明記しています。95%区間は凍結評価器のペア用関数（名称はpaired_wilson_ci95、実体はペア得点の切り下げ／切り上げによるClopper–Pearson区間）。差の区間は同じ初期乱数の自然終了共通ペアを単位とするbootstrap参考値です。

## 何を直して学習したか

元モデルが自分のルキナのサーチへ灰流うららを発動する21操作の記録を再現しました。カードID・操作番号・発動種別の対応は正しく、この再現で取り違えは確認されませんでした。旧入力には処理待ちチェーンの発動者とカードを明示する項目がありませんが、自分の操作履歴や盤面から間接的に推測する余地はあります。入力の不足が唯一の原因だと断定しません。

公開MSG_CHAINING通知からチェーン長・直近の発動者・カード・効果番号を取り出す実験環境と、小さな学習可能な補助ネットワークを作りました。自己発動を禁止する固定ルールは入れていません。相手の非公開手札・デッキのIDはモデルに渡していません。状態確認用の内部盤面は正解の判定にのみ使います。

実エンジンで実行できた手順を、学習180・検証34・最終評価66軌跡に分離しました。展開の初手は添え札の枚数構成ごとに分け、同じ手札の並び替えが別分割へ入らないようにしました。既存24ケースも学習・検証に含めていません。8種の展開に対する別初手48ケースであり、48種類の独立した展開パターンではありません。2種は途中盤面からの合成ケースです。

学習はGPUで600更新、バッチ4、元モデル学習率3e-5・補助機能3e-4、元方策へのKL係数0.01。元の69,632判断の強化学習に、今回の手本学習を追加しました。600更新を追加の強化学習判断数とは数えていません。検証データの損失だけでstep {tr['selected_step']}を選び、最終評価・対戦結果で選び直していません。全重みは有限、元の重みのハッシュは変化していません。

手本は、短い目標盤面へ到達できることをエンジンで確認したものです。通常召喚権・指定した添え札の温存なども採点条件に含めるため、別の有効な戦略が不合格になる場合もあります。同じ8種の基本展開を別の添え札で試しており、未知の展開・未知デッキへの一般化を示す試験ではありません。発動判断で追加したミクス・トラックメイカーも、展開手本には登場するカードです。

人間の熟練者による最適性の確認はまだです。相手は展開試験では見送り、発動判断試験では直前まで固定手順です。妨害を受けた再展開、妨害の温存、長いチェーン、ターン終了時の最適盤面、他デッキのカード効果全般はこの満点で保証されません。KLによる元方策の維持も手本の盤面上での制約なので、対戦全体の戦い方を維持する保証にはなりません。

## 保存場所と再現条件

- WSL実験本体: `{B}`
- 選択した重み: `{selected}`（baseとchain-headの両方）
- 読める操作列: `evaluation/モデル名/holdout/*.md` と `chain_holdout/*.md`
- 学習分割と手本: `dataset/split-manifest.json` と `*-witnesses.json`
- 対戦原本: `battle/モデル名/相手名/summary.json`、`episodes.csv`
- 推論の必須条件: `model/runtime-contract.json`
- 実行用ソース: `reproduction/`、ネイティブ変更: `runtime-source/public-chain.patch`

新モデルは実験用として別保存し、既存の評価器・元モデル・通常環境は上書きしていません。通常の旧CLIにbaseファイルだけ渡すと今回の候補と異なる条件になります。推論には公開チェーン情報を返す実験環境とbase＋chain-headを組み合わせます。ソースはこのPCの既存WSLパスと固定依存環境を参照します。別PCへコピーするだけで動く配布物ではありません。

通常対戦16戦の回帰確認は勝敗・操作数を含め元環境と一致し、評価のLuaエラーは0、各ジョブ終了時の残存デュエルは0でした。実行プロセスには5GiB上限を設定し、成功ジョブの実測最大メモリは約{peak:.0f}MiB、学習時GPU全体の最大使用量は{tr['peak_total_gpu_memory_mib']:.0f}MiBでした。

評価を読み込む際のJSON整数キー、対戦ドライバの実験パスの指定を修正して再実行しています。学習重み・データ内容・目標判定・最終評価の選択方針は変更していません。
'''
(O/'README.md').write_text(report)
(B/'report.md').write_text(report)
(P/'docs/chain-imitation-2026-11-08.md').write_text(report)
index=P/'README.md';text=index.read_text();link='docs/chain-imitation-2026-11-08.md'
if link not in text:index.write_text(text+'\n\n## 2026-11-08 展開の手本学習\n\n結果と適用範囲は[展開・発動判断の検証報告](docs/chain-imitation-2026-11-08.md)を参照してください。新モデルは別保存した実験用です。\n')
checks=dict(status='passed',csv_matches_raw_json=True,source_checkpoint_unchanged=True,user_scoring_evaluator_unchanged=True,
            dataset_manifest_unchanged=True,selected_by_validation_not_holdout=True,baseline_diagnostic_reproduced=True,
            normal_runtime_regression_16_games_passed=True,total_battle_games=512,abnormal_battle_games=abnormal,
            lua_errors=0,live_duels_after_each_job=0,package_file_hashes_verified=False)
write(O/'verification.json',checks)
files={str(p.relative_to(O)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SHA256-manifest.json'}
write(O/'SHA256-manifest.json',files)
print(json.dumps(dict(output=str(O),files=len(files),abnormal=abnormal,battle_pairs=paired),ensure_ascii=False))
