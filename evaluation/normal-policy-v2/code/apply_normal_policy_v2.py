import json,hashlib,shutil
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');R=P/'evaluation/normal-policy-v2'
W=Path(__file__).parent;C=R/'code';C.mkdir(parents=True,exist_ok=True)
names=['chain_imitation_common.py','chain_policy.py','inspect_full_turn.py','endboard_witness.py','verify_user_rukina_reference.py',
       'diagnose_user_rukina_model.py','verify_normal_policy_v2.py','apply_normal_policy_v2.py']
for name in names:shutil.copy2(W/name,C/name)
policy=dict(version='normal-summon-agnostic-v2',normal_summon_usage_scored=False,
    normal_summon_usage_is_diagnostic_only=True,require_exact_normal_summon_count=False,
    legal_summon_limits_enforced_by_engine=True,normal_preservation_bonus_or_penalty=False,
    user_instruction='召喚権温存という条件をやめる',
    historical_results='Old reports, frozen reproduction code, weights and corpora retain their original provenance; not current acceptance criteria.',
    excluded_from_automatic_training_replay=['experiments/chain-imitation-20261108/dataset','experiments/endboard-20261108/dataset'])
(R/'policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
doc='''# 現行の展開評価方針（2026-11-08、ユーザー指示）

「召喚権を使わず残す」という必須条件は撤廃しました。
通常召喚・セットの回数は観察用に記録するだけで、合否・加点・減点に使いません。「ちょうど1回」の固定条件も付けません。
実際に召喚できるか、追加の召喚が許されるかはルールエンジンが判定します。

現行コードはscripts/eval-combo-benchmark.pyとevaluation/normal-policy-v2/code以下です。
従来の盤面・対象選択などの判定を、召喚回数に依存しない形へ修正しました。
ユーザー指定のルキナ1枚展開では、獄神精を通常召喚して到達する盤面を正しく合格にできます。

過去の実験結果・凍結した再現用コードは旧条件の履歴です。数値を現行条件で測ったものへ書き換えず、現在の正答率として流用しません。
旧条件で作成した手本は自動で再学習用へ混ぜません。旧モデルに残る学習の影響は、評価コードの修正だけでは消えないため、条件撤廃済みのモデルとして扱いません。

検証結果：evaluation/normal-policy-v2/verification.json。
'''
(P/'docs/evaluation-policy-current.md').write_text(doc)
idx=P/'README.md';text=idx.read_text()
notice='''## 現行の評価方針（2026-11-08更新）

召喚権温存の必須条件を撤廃しました。通常召喚・セット回数は記録のみで、合否や加減点には使いません。
[現行方針](docs/evaluation-policy-current.md)を参照してください。以下の旧実験の温存条件と成功率は履歴であり、現行基準ではありません。

'''
if notice not in text:
    first,rest=text.split('\n',1);text=first+'\n\n'+notice+rest.lstrip('\n')
text=text.replace('詳細と手順は `docs/combo-benchmark-2026-11-07.md`、再実行は `scripts/eval-combo-benchmark.py` を参照してください。',
    '旧条件の詳細は `docs/combo-benchmark-2026-11-07.md`、旧結果の再現用コードは `experiments/combo-benchmark-20261107/evaluation-v1/runner.py` です。`scripts/eval-combo-benchmark.py` は現行条件へ更新済みです。')
idx.write_text(text)
banner='> **旧評価条件の履歴です。召喚権温存の条件は撤廃済みです。現行方針は[evaluation-policy-current.md](evaluation-policy-current.md)を参照してください。過去の数値を現行の正答率として流用しません。**\n\n'
for name in ['combo-benchmark-2026-11-07.md','chain-imitation-2026-11-08.md','endboard-2026-11-08.md','user-rukina-reference-2026-11-08.md']:
    p=P/'docs'/name;t=p.read_text()
    if not t.startswith(banner):p.write_text(banner+t)
(P/'scripts/check-expansion-grading.py').write_text("import sys,runpy\nfrom pathlib import Path\np=Path('/home/hatk8/projects/yugioh-ai/evaluation/normal-policy-v2/code')\nsys.path.insert(0,str(p))\nrunpy.run_path(str(p/'verify_normal_policy_v2.py'),run_name='__main__')\n")
(R/'code-hashes.json').write_text(json.dumps({n:hashlib.sha256((C/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n')
print('APPLIED_NORMAL_SUMMON_AGNOSTIC_POLICY')
