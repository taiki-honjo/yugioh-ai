"""Register every accessible article replay occurrence and text-only opening/branch."""
import copy,csv,hashlib,json,shutil,sqlite3,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from all_opponent_reference_inventory import *
P=Path('/home/hatk8/projects/yugioh-ai');W=Path(__file__).parent
D=P/'records/expansion-references-20261108';O=W.parent/'outputs/expansion-policy-2026-11-08'
D.mkdir(exist_ok=True);O.mkdir(exist_ok=True)
current=D/'references.json';backup=D/'references-partial-v1.json'
if not backup.exists():shutil.copy2(current,backup)
old=json.loads(backup.read_text());previous=copy.deepcopy(old);sources={x['deck']:x for x in previous['sources']}
run=Path((P/'experiments/meta-opponents-20261107/latest-run.txt').read_text().strip());meta=json.loads((run/'manifest.json').read_text())
recipes={x['key']:x for x in meta['recipes']}
db=sqlite3.connect(P/'experiments/chain-imitation-20261108/runtime/agent/assets/locale/ja/cards.cdb')
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def name(code):
    row=db.execute('select name from texts where id=?',(code,)).fetchone();return row[0] if row else str(code)
def starters(deck,label):
    if deck=='ClownTroupe':
        mapping={'フレア':42759961,'ハット':19491080,'極芸':83232904,'下稽古':20448151,'開演':57847269,
                 'ホワイトフェイス':82159583,'フェンリル':32909498}
        return list(dict.fromkeys(code for term,code in mapping.items() if term in label))
    if label=='襲撃':return [1122030]
    if label.startswith('メディウス'):return [97556336]
    if label.startswith('獄神獣'):return [23829452]
    if label.startswith('学都'):return [74733322]
    if label=='テルミナス':return [25661743]
    return []
def entry(deck,label,kind,category,anchor,url,entry_id,codes=None,starter_text=None):
    article=CLONE if deck=='ClownTroupe' else ART;cards={x['code']:x for x in recipes[deck]['entries']}
    starter_text=label if starter_text is None else starter_text
    codes=starters(deck,starter_text) if codes is None else codes
    required=[dict(code=k,name=name(k),selected_count=cards[k]['count'] if k in cards else 0,present=k in cards) for k in codes]
    missing=[x for x in required if not x['present']]
    e=dict(id=entry_id,label=label,reference_type=kind,category=category,url=url,
        source_section_url=article+'#'+anchor,source_heading_id=anchor,starter_codes=codes,starter_cards=required,
        registration_status='registered',normal_summon_usage_scored=False,
        validation=dict(article_reference_observed=True,full_replay_reviewed=False,local_engine_verified=False,used_for_training=False),
        initial_named_starters_in_selected_recipe=not missing if codes else None,
        compatibility=dict(missing_named_starters=missing,full_route_requirements_checked=False,
            status='named_starter_missing_in_selected_recipe' if missing else 'named_starters_present_full_route_pending' if codes else 'initial_state_pending_replay_or_witness_review'),
        status='registered_reference_only; full local-engine replay pending')
    if missing:e['training_eligible']=False
    if 'コスト' in label:e['additional_start']='記事にあるコストの種類・枚数を別条件として維持し、動画の開始条件と照合する。'
    categories=[t for t in ['レベル3チューナー','下級','最上級'] if t in starter_text]
    if '上級' in starter_text and '最上級' not in starter_text:categories.append('上級')
    if categories:e['generic_starter_requirements_from_article']=categories
    return e
cl=[]
for slug,label,section,heading in CL_ROWS:
    e=entry('ClownTroupe',label,'replay',section,CL_ANCHORS[heading],BASE+slug,'clown-'+slug,starter_text=heading)
    e['source_heading']=heading;cl.append(e)
cl_by_slug={x['url'].rsplit('/',1)[1]:x for x in cl}
cl_by_slug['a56juEEA']['article_branch_notes']=[
    '妨害を示す反応がない場合のホワイトフェイス確保。',
    '反応がある場合のホワイトフェイス通常召喚。',
    '極芸がうららで止まった場合の下稽古への接続。']
for slug in ['kbytB4Ps','LvB4Tdeq','aRq7xmEC','kLgRsQfa']:
    cl_by_slug[slug]['article_observation_note']='記事の分岐は対戦画面の反応を前提にしている。相手の非公開手札をモデル入力に追加しない。'
cl_by_slug['3k9GGRWL']['article_branch_notes']=['ドロールを止める札を持つ条件と墓地のリンクリボーを重視する形。']
cl_by_slug['qLGavj6P']['article_branch_notes']=['フレア以外に下稽古へ払えるコストを持つ場合。']
cl_by_slug['LNA98bpm']['article_branch_notes']=['別のコストがない場合と、うららを意識した順序。']
for slug in ['u7GBEa4h','XBJXaFeM']:
    cl_by_slug[slug]['article_branch_notes']=['ハットの効果が止められなかった場合を前提とする。']
sources['ClownTroupe']['routes']=cl
sources['ClownTroupe']['coverage']=dict(all_public_replay_occurrences_registered=True,observed_replay_occurrences=32,
    observed_unique_replay_urls=32,registered_replay_occurrences=len(cl),registered_unique_replay_urls=len({x['url'] for x in cl}),
    source_heading_count=len(CL_ANCHORS),registered_source_heading_count=len({x['source_heading_id'] for x in cl}),
    category_counts=dict(Counter(x['category'] for x in cl)),public_openings_and_branches_complete=True,
    paid_appendix_accessible=False,excluded_sections=[dict(label='有料Appendix',reason='未購入のため本文未取得。登録済み・確認済みとして扱わない。')])
sources['ClownTroupe']['scope']='公開部分の全初動・分岐と32本の既存リプレイ。有料Appendixは対象外。'
art=[];occurrences=Counter()
for slug,label,category,anchor in AR_ROWS:
    occurrences[slug]+=1;e=entry('Artmage',label,'replay',category,anchor,BASE+slug,f'artmage-{slug}-{occurrences[slug]}')
    if slug=='wsMMwXxE':
        e['cost_kind']='monster' if 'モンスターコスト' in label else 'spell_or_trap'
        e['starter_alternatives']=[[97556336]] if e['cost_kind']=='monster' else [[97556336],[1122030]]
        e['draw_result_not_fixed_for_training']=True
    if slug=='dAvrxdzV':e['article_branch_notes']=['無効系を意識した追加の通常召喚を使う分岐も参照する。']
    if slug=='qRShaDNM':e['article_branch_notes']=['ネルヴェド2枚を用いる追加の形は、現在の1枚採用と区別して検証する。','メディウスとコストを持つ場合の別案も本文にある。']
    if slug=='4FK42xFG':e['article_branch_notes']=['テルミナス以外に、必要なコストがあるメディウス・学都の始動も本文で言及されている。']
    art.append(e)
for key,label,anchor,note in AR_TEXT:
    e=entry('Artmage',label,'article_text','誘発ケア',anchor,None,'artmage-text-'+key)
    e['article_branch_notes']=[note];e['status']='registered_text_only_branch; engine witness pending';art.append(e)
notes=[]
for key,label,anchor,codes,note in AR_OPENING_NOTES:
    e=entry('Artmage',label,'opening_note','カード紹介中の補足初動',anchor,None,'artmage-note-'+key,codes)
    e['article_branch_notes']=[note];e['status']='registered_opening_note; full sequence not provided in article';notes.append(e)
sources['Artmage']['routes']=art;sources['Artmage']['opening_notes']=notes
sources['Artmage']['context_notes']=[dict(label='将来の採用候補の記述',source_section_url=ART+'#5929eada-109e-461f-bd40-35c1ccd20c23',
    note='ビフレストに関する記述は将来候補の説明であり、現在の採用済みレシピへ勝手に追加しない。')]
sources['Artmage']['coverage']=dict(all_public_replay_occurrences_registered=True,observed_replay_occurrences=13,
    observed_unique_replay_urls=12,registered_replay_occurrences=sum(x['reference_type']=='replay' for x in art),
    registered_unique_replay_urls=len({x['url'] for x in art if x['url']}),basic_opening_cases=7,
    interruption_cases=8,text_only_branches=2,supplementary_opening_notes=len(notes),
    duplicate_replay_url_conditions_preserved=True,public_openings_and_branches_complete=True,full_article_accessible=True)
sources['Artmage']['scope']='公開記事にある基本7条件、誘発対応8条件と補足初動4件。リプレイは12本、掲載箇所は13件。'
data=dict(schema_version=2,status='all_public_opening_references_registered',grading_policy='normal-summon-agnostic-v2',
    normal_summon_usage_scored=False,publish_new_replays=False,training_from_these_articles_completed=False,
    registration_scope='Accessible public article openings and branches; references, not completed engine witnesses.',
    sources=[sources['ClownTroupe'],sources['Artmage']],validation_order=old['validation_order'])
all_entries=cl+art+notes
proof_path=P/'evaluation/opponent-article-basics-20261108/verification-index.json'
proof_index=json.loads(proof_path.read_text()) if proof_path.exists() else dict(entries={})
for e in all_entries:
    proof=proof_index['entries'].get(e['id'])
    if proof:
        e['validation'].update(local_engine_verified=proof['local_engine_verified'],full_replay_reviewed=proof['full_replay_reviewed'])
        e['validation']['used_for_training']=proof.get('used_for_training',False)
        e['engine_witness']=proof
        e['engine_witness_report']=proof_index['report_windows_path']
        e['status']='selected-recipe opening fine-tuned; other routes not trained' if e['validation']['used_for_training'] else 'selected-recipe pilot witness verified with documented conditions; not trained'
for source in old['sources']:
    before_urls={x['url'] for x in source['routes']};after_urls={x['url'] for x in sources[source['deck']]['routes'] if x['url']}
    assert before_urls<=after_urls,(source['deck'],before_urls-after_urls)
assert len(cl)==32 and len({x['url'] for x in cl})==32
assert len(art)==15 and len(notes)==4 and len({x['url'] for x in art if x['url']})==12
assert [x['id'] for x in all_entries]==list(dict.fromkeys(x['id'] for x in all_entries))
assert all(x['normal_summon_usage_scored'] is False and x['registration_status']=='registered' for x in all_entries)
assert {x['url'].rsplit('/',1)[1] for x in cl}==set(EXPECTED_CL_SLUGS)
assert {x['url'].rsplit('/',1)[1] for x in art if x['url']}==set(EXPECTED_AR_SLUGS)
coverage=dict(status='passed',all_44_public_replay_urls_registered=True,all_45_public_replay_occurrences_registered=True,
    no_previously_registered_url_lost=True,clown_routes=32,artmage_cases=15,artmage_supplementary_opening_notes=4,
    total_reference_entries=51,text_only_branches=2,distinct_replay_urls=44,source_link_occurrences=45,
    excluded_scope='ClownTroupe paid Appendix',engine_replay_verified=False,trained_from_references=False,
    engine_verified_reference_entries=sum(e['validation']['local_engine_verified'] for e in all_entries),
    trained_reference_entries=sum(e['validation']['used_for_training'] for e in all_entries),all_reference_entries_trained=False,
    all_reference_entries_engine_verified=False,
    missing_named_starter_entries=[dict(id=x['id'],label=x['label'],missing=x['compatibility']['missing_named_starters']) for x in all_entries if x['compatibility']['missing_named_starters']])
data['coverage_verification']=coverage
for folder in [D,O]:
    write(folder/'references.json',data);write(folder/'reference-coverage.json',coverage)
    with (folder/'references.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=['deck','id','category','label','reference_type','replay_url','source_section_url','missing_named_starters','local_engine_verified','used_for_training'])
        writer.writeheader()
        for deck,entries in [('ClownTroupe',cl),('Artmage',art+notes)]:
            for x in entries:writer.writerow(dict(deck=deck,id=x['id'],category=x['category'],label=x['label'],reference_type=x['reference_type'],
                replay_url=x['url'] or '',source_section_url=x['source_section_url'],
                missing_named_starters=';'.join(k['name'] for k in x['compatibility']['missing_named_starters']),local_engine_verified=x['validation']['local_engine_verified'],used_for_training=x['validation']['used_for_training']))
lines=['# 両記事の全公開初動・分岐の参照一覧','',
    '登録は、記事にある見出し・条件・既存リプレイへの参照を一覧に保持した状態です。各条件の再現確認とAIへの学習投入は別工程です。','',
    '| 記事 | 登録した条件・分岐 | 補足初動 | 異なるリプレイURL数 |','|---|---:|---:|---:|',
    '| 道化の一座 | 32 | 0 | 32 |','| アルトメギア | 15 | 4 | 12 |','',
    'アルトメギアは同じメディウスの動画を使う2種類のコストを別条件で保持しています。動画に対応する掲載箇所は合計45件、異なるURLは44本です。動画がない無効系・幽鬼うさぎの説明も、文章の分岐として登録しました。','',
    '道化の一座の有料Appendixは未取得で、網羅済みとは扱いません。公開部分にある初動と分岐はすべて登録しました。','',
    'フェンリルを使う道化の一座の3件は、現在の採用済みレシピにフェンリルがないため、その不一致を残して登録しました。参照を削除したり、デッキを勝手に変更したりはしていません。','',
    '召喚権温存の必須条件は使いません。新しいリプレイのサーバー保存・再公開も行いません。','']
if proof_index['entries']:
    lines+=['フレア始動と襲撃始動の2件は、採用済みレシピで条件を明示した再現確認を各3回完了しました。全51件の検証やAIへの学習投入が完了した意味ではありません。',
        f"[盤面画像と確認条件]({proof_index['report_windows_path']})",'']
if proof_index.get('trained_from_references'):
    lines+=['上記2初動は各デッキの対戦相手AIへ追加学習しました。全51件を学習した状態ではありません。',
        f"[追加学習の結果とモデル]({proof_index['training_report_windows_path']})",'']
for key,entries in [('ClownTroupe',cl),('Artmage',art)]:
    source=sources[key];lines += ['## '+source['label'],'',f"[{source['title']}]({source['article']})",'']
    for category in dict.fromkeys(x['category'] for x in entries):
        lines += ['### '+category,'','| ID | 条件・分岐 | 参照先 | 照合事項 |','|---|---|---|---|']
        for x in entries:
            if x['category']!=category:continue
            link=f"[リプレイ]({x['url']})" if x['url'] else f"[本文]({x['source_section_url']})"
            caveat=('当初動の追加学習済み（詳細参照）' if x['validation']['used_for_training'] else '条件を明示した再現確認3回完了・未学習' if x['validation']['local_engine_verified'] else
                '、'.join(k['name']+'未採用' for k in x['compatibility']['missing_named_starters']) or ('開始手札は動画確認待ち' if not x['starter_codes'] else '全手順の確認待ち'))
            lines.append(f"| {x['id']} | {x['label']} | {link} | {caveat} |")
        lines.append('')
    if key=='Artmage':
        lines += ['### カード紹介中に言及された補足初動','','| ID | 初動の補足 | 記事の該当箇所 |','|---|---|---|']
        for x in notes:lines.append(f"| {x['id']} | {x['label']} | [本文]({x['source_section_url']}) |")
        lines+=['','補足初動は記事内の言及を登録したものです。掲載されていない全操作やコストを創作して、検証済みの手順とは扱いません。','']
lines += ['機械可読の一覧：references.json / references.csv。網羅確認：reference-coverage.json。','']
report='\n'.join(lines)
(O/'README.md').write_text(report);(P/'docs/expansion-references-2026-11-08.md').write_text(report);(D/'README.md').write_text(report)
for name in ['register_all_opponent_references.py','all_opponent_reference_inventory.py']:
    shutil.copy2(W/name,O/name);shutil.copy2(W/name,D/name)
# Keep the old entry point from silently restoring the former 12-link subset.
legacy=W/'register_opponent_references.py'
legacy.write_text("\"\"\"Compatibility entry point: register the complete public inventory.\"\"\"\nimport runpy\nfrom pathlib import Path\nrunpy.run_path(str(Path(__file__).with_name('register_all_opponent_references.py')),run_name='__main__')\n")
shutil.copy2(legacy,O/legacy.name)
manifest={str(p.relative_to(O)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in O.rglob('*') if p.is_file() and p.name!='SHA256-manifest.json'}
write(O/'SHA256-manifest.json',manifest)
print(json.dumps(dict(output=str(O),coverage=coverage,files=len(manifest)),ensure_ascii=False))
