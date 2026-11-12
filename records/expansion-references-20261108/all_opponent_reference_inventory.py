"""Public article link/heading inventory observed in the browser on 2026-11-08.

Only link metadata and concise condition descriptions are recorded, not article bodies.
"""
BASE='https://duel-simulator.miyauchidp.dev/replay/'
CLONE='https://note.com/fugapoke/n/n444a2760344c'
ART='https://note.com/kind_borage903/n/n0b8910935f9a'
# One entry per visible replay-link occurrence, preserving different conditions.
CL_ROWS=[
 ('CRWQjPaN','フレア1枚・ホワイトフェイス残し','1枚初動','フレア1枚'),
 ('Qpqm2Dm7','ハット1枚・基本','1枚初動','ハット1枚'),
 ('jD6qEYbq','ハット1枚・フレア用の手札コストなし','1枚初動','ハット1枚'),
 ('jcUpCPpQ','下稽古＋モンスターコスト','2枚初動','下稽古 + モンスターコスト'),
 ('FDcws6wr','極芸＋レベル3チューナー','2枚初動','極芸 + レベル3チューナー'),
 ('5JRKg6xw','極芸＋下級','2枚初動','極芸 + 下級'),
 ('mQWBQNdJ','極芸＋上級','2枚初動','極芸 + 上級'),
 ('wuH3zLrP','極芸＋最上級','2枚初動','極芸 + 最上級'),
 ('CAaV2Y29','極芸＋フェンリル・ドロールを意識した形','2枚初動','極芸 + フェンリル'),
 ('3k9GGRWL','極芸＋フェンリル・墓地リンクリボーを優先','2枚初動','極芸 + フェンリル'),
 ('Qh9hpLGk','極芸＋開演','2枚初動','極芸 + 開演'),
 ('neYwJkTf','開演＋レベル3チューナー','2枚初動','開演 + レベル3チューナー'),
 ('YhJYVhES','開演＋上級・コスト別','2枚初動','開演 + 上級'),
 ('xmseE4Kw','開演＋上級・下稽古のコストなし','2枚初動','開演 + 上級'),
 ('q4KANa5q','開演＋フェンリル','2枚初動','開演 + フェンリル'),
 ('kbytB4Ps','フレア＋ハット・うららを意識した分岐','誘発ケア展開','フレア + ハット'),
 ('dWbA4y9a','フレア＋ハット・下稽古にうららを受けた後','誘発ケア展開','フレア + ハット'),
 ('LvB4Tdeq','フレア＋ハット・ドロールを意識した分岐','誘発ケア展開','フレア + ハット'),
 ('KrCwwyjK','フレア＋ハット・ドロールを受けた後','誘発ケア展開','フレア + ハット'),
 ('a56juEEA','フレア＋極芸','誘発ケア展開','フレア + 極芸'),
 ('qLGavj6P','フレア＋下稽古・別のコストあり','誘発ケア展開','フレア + 下稽古'),
 ('LNA98bpm','フレア＋下稽古・別のコストなし','誘発ケア展開','フレア + 下稽古'),
 ('A8hYg7NE','フレア＋ホワイトフェイス','誘発ケア展開','フレア + 最上級（ホワイトフェイス）'),
 ('BqbjaHU9','ハット＋極芸','誘発ケア展開','ハット + 極芸'),
 ('AMJhX78m','ハット＋下稽古','誘発ケア展開','ハット + 下稽古'),
 ('u7GBEa4h','ハット＋開演','誘発ケア展開','ハット + 開演'),
 ('XBJXaFeM','ハット＋上級','誘発ケア展開','ハット + 上級'),
 ('aRq7xmEC','下稽古＋開演・ディアボロを使う分岐','誘発ケア展開','下稽古 + 開演'),
 ('kLgRsQfa','下稽古＋開演・ハットを使う分岐','誘発ケア展開','下稽古 + 開演'),
 ('2Yh4gcjq','下稽古＋極芸','誘発ケア展開','下稽古 + 極芸'),
 ('m4J8mjmq','フレア1枚・ドロー枚数優先','その他展開','フレア1枚（ドロー枚数優先）'),
 ('BSb6759k','極芸を止められた後の下稽古','その他展開','極芸が止められた後の下稽古'),
]
CL_ANCHORS={
 'フレア1枚':'299f34f6-921e-40cb-b0df-59bed99ead72',
 'ハット1枚':'4dc0dba1-24f2-47c1-b283-993a83d87a13',
 '下稽古 + モンスターコスト':'5b60fba8-195e-45f5-b03c-77718c3c22d1',
 '極芸 + レベル3チューナー':'d9edf3e7-7c0e-4a20-9d4a-ae017b88558d',
 '極芸 + 下級':'4d6502f2-1943-4e0c-948b-ade612ae9c5d',
 '極芸 + 上級':'0d411b06-05da-4be9-95cb-19b32059401f',
 '極芸 + 最上級':'8b2402cc-e01d-498b-a257-f20ab82798c8',
 '極芸 + フェンリル':'09458a20-cfde-4a78-b201-eff92ce4f742',
 '極芸 + 開演':'3d318413-f1ee-4e82-9618-f178ea8ca08f',
 '開演 + レベル3チューナー':'87240159-ba16-4fc4-92e8-98a72fc4ba0b',
 '開演 + 上級':'bde50204-1ee2-45e4-9d6a-6bfcbd4f91bc',
 '開演 + フェンリル':'7a6287ee-9539-450c-a062-2319e1df231b',
 'フレア + ハット':'cf524049-fa66-4b17-be8b-d0bfd973f655',
 'フレア + 極芸':'f1d882f0-040f-4e46-9987-947902f0068d',
 'フレア + 下稽古':'85c6a814-cd6d-4651-b58a-526d2ef65f98',
 'フレア + 最上級（ホワイトフェイス）':'0de31487-adae-4525-ac71-95447abef8c5',
 'ハット + 極芸':'9ce9717b-a571-43f7-b594-d99422682262',
 'ハット + 下稽古':'3ad8b04b-62b9-4f91-91b7-a13a914ae273',
 'ハット + 開演':'81fac8f4-b814-4508-8f09-08b55c7e90ea',
 'ハット + 上級':'c418a6ed-954f-41ef-8289-6233fb51a8b2',
 '下稽古 + 開演':'54264f29-0a9f-4670-8ee9-41821bbe67ef',
 '下稽古 + 極芸':'3830425c-bbb3-4e61-b4fc-df436a84f7c9',
 'フレア1枚（ドロー枚数優先）':'05ab4fa6-90b5-4a35-9c6c-71ef46208aee',
 '極芸が止められた後の下稽古':'5762b115-c1e8-4ed2-8f4b-eceb018ec7e7',
}
AR_ROWS=[
 ('g2rN57Xb','襲撃','基本展開','0243b613-1cf3-4e82-9dd1-85c08ad52809'),
 ('wsMMwXxE','メディウス＋モンスターコスト','基本展開','eb14d276-bbd1-4b10-b978-45a83b477c8d'),
 ('wsMMwXxE','メディウス／襲撃＋魔法罠コスト','基本展開','042c7be4-135d-45f2-bc48-4a12c7aa0499'),
 ('vxs2HMAV','獄神獣初動','基本展開','eafda685-f366-49bd-bcdb-59a94feb096f'),
 ('XUS4azXa','学都＋魔法罠コスト1枚','基本展開','426adae2-a1f9-406a-a4fa-5b53dac92499'),
 ('dAvrxdzV','学都＋魔法罠コスト2枚','基本展開','e476c29e-eeb5-4a74-8999-67a65e2e4c64'),
 ('xXerEKuW','テルミナス','基本展開','a3d1c80c-039e-430c-a4f0-b638bbf09f8e'),
 ('rYcfTsW2','調和ノ天救竜への対応','誘発ケア','ffb779c2-a614-468e-b4e4-bda5a7df6cc0'),
 ('XG5YrpJk','マルチャミー・フワロスへの対応','誘発ケア','b6c92f38-cc7d-4c43-b4b7-e0c48cfe25f3'),
 ('QaCTRqBB','増殖するGへの対応','誘発ケア','29415ab2-12d4-4dd3-aad3-c3d34f944b0d'),
 ('qRShaDNM','ドロールへの対応','誘発ケア','fb999127-4ea0-48c5-9271-5a41050bbb58'),
 ('dBmNcMDR','レギナを通された場合の対応','誘発ケア','19b382c8-7554-4027-a0c6-048149ff693e'),
 ('4FK42xFG','マルチャミー・プルリアへの対応','誘発ケア','66017f3e-53e8-4f7b-8772-8f01362de71a'),
]
AR_TEXT=[
 ('negation','無効系への対応','a52b63ba-eccf-4925-9804-19a5dbc7be91','獄神獣を使う回避や、無効になった個体を素材にする案。完成した個別手順ではない。'),
 ('ogre','幽鬼うさぎへの対応','59d3832a-c6f6-469b-96bc-66224ffdb584','学都の効果を使う順番と、改変を併せ持つ場合や奪還・襲撃を使う案を分ける。'),
]
AR_OPENING_NOTES=[
 ('varnish','改変からの初動接続','77bf4f0e-b9ba-48fd-b991-043537a943cb',[74011784],'初動にアクセスするカードとしての説明。専用の全手順は未掲載。'),
 ('finemelt','ファインメルトから知恵の乙女を使う初動','566e36f5-8223-4301-a226-911135128bb6',[34541940],'初動として言及あり。必要な追加コストと全手順は今後確認する。'),
 ('nervedo','ネルヴェドから知恵の乙女を使う初動','566e36f5-8223-4301-a226-911135128bb6',[17473466],'初動として言及あり。必要な追加コストと全手順は今後確認する。'),
 ('beast-finemelt','獄神獣の通常召喚＋ファインメルト','890b7fb0-e727-4493-98ce-1e9edd0beae2',[23829452,34541940],'カード紹介内の2枚の動き。専用のリプレイはない。'),
]
# Independent compact inventories copied from the public DOM link survey.
EXPECTED_CL_SLUGS='CRWQjPaN Qpqm2Dm7 jD6qEYbq jcUpCPpQ FDcws6wr 5JRKg6xw mQWBQNdJ wuH3zLrP CAaV2Y29 3k9GGRWL Qh9hpLGk neYwJkTf YhJYVhES xmseE4Kw q4KANa5q kbytB4Ps dWbA4y9a LvB4Tdeq KrCwwyjK a56juEEA qLGavj6P LNA98bpm A8hYg7NE BqbjaHU9 AMJhX78m u7GBEa4h XBJXaFeM aRq7xmEC kLgRsQfa 2Yh4gcjq m4J8mjmq BSb6759k'.split()
EXPECTED_AR_SLUGS='g2rN57Xb wsMMwXxE vxs2HMAV XUS4azXa dAvrxdzV xXerEKuW rYcfTsW2 XG5YrpJk QaCTRqBB qRShaDNM dBmNcMDR 4FK42xFG'.split()
assert len(CL_ROWS)==32 and {x[0] for x in CL_ROWS}==set(EXPECTED_CL_SLUGS)
assert len(AR_ROWS)==13 and {x[0] for x in AR_ROWS}==set(EXPECTED_AR_SLUGS)
