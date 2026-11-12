import hashlib,json,shutil,subprocess
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');B=P/'experiments/chain-imitation-20261108';B.mkdir(parents=True,exist_ok=True)
old=P/'experiments/combo-benchmark-20261107/runtime';R=B/'runtime';original=P/'experiments/elfnotes-20261106/runtime'
if not R.exists():
    R.mkdir();shutil.copytree(old/'agent',R/'agent',symlinks=True,ignore=shutil.ignore_patterns('*.so','__pycache__','.git'))
    for name in ['core','include','fmt','sqlitecpp','unordered-dense','card-scripts','effect_checks.h']:
        (R/name).symlink_to(original/name,target_is_directory=(original/name).is_dir())
h=R/'agent/ygoenv/ygoenv/ygopro/ygopro.h';t=(old/'agent/ygoenv/ygoenv/ygopro/ygopro.h').read_text()
needle='  PlayerId chaining_player_;';assert t.count(needle)==1
t=t.replace(needle,needle+'\n  std::vector<std::array<uint32_t,3>> public_chain_; // code, activating player, public description\n')
needle='    disclosures_.clear();\n';assert needle in t
t=t.replace(needle,needle+'    public_chain_.clear();\n',1)
needle='        "info:termination_reason"_.Bind(Spec<int>({}, {0, 2})),\n';assert needle in t
t=t.replace(needle,needle+'''        "info:public_chain_length"_.Bind(Spec<int>({})),
        "info:public_chain_player"_.Bind(Spec<int>({})),
        "info:public_chain_card_id"_.Bind(Spec<int>({})),
        "info:public_chain_effect"_.Bind(Spec<int>({})),
''')
needle='    state["info:termination_reason"_] = termination_reason_;';assert t.count(needle)==1
t=t.replace(needle,needle+'''
    state["info:public_chain_length"_] = int(public_chain_.size());
    state["info:public_chain_player"_] = public_chain_.empty() ? -1 : int(public_chain_.back()[1]);
    int chain_cid=0;
    if(!public_chain_.empty()) {
      auto found=card_ids_.find(public_chain_.back()[0]);
      if(found!=card_ids_.end()) chain_cid=found->second;
    }
    state["info:public_chain_card_id"_] = chain_cid;
    state["info:public_chain_effect"_] = public_chain_.empty() ? 0 : int(public_chain_.back()[2]&15);
''')
needle='''    } else if (msg_ == MSG_CHAIN_END) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      auto tc = read_u8();
      auto tl = read_u8();
      auto ts = read_u8();
      uint32_t desc = read_u32();
      auto cs = read_u8();'''
assert t.count(needle)==1
t=t.replace(needle,'''    } else if (msg_ == MSG_CHAIN_END) {
      public_chain_.clear();
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINING) {
      CardCode code = read_u32();
      auto public_location = read_u32();
      auto tc = read_u8();
      auto tl = read_u8();
      auto ts = read_u8();
      uint32_t desc = read_u32();
      auto cs = read_u8();
      public_chain_.resize(cs>0 ? cs-1 : 0);
      public_chain_.push_back({code & 0x7fffffff, uint32_t(tc), desc});
      if (!verbose_) {dp_=dl_;return;}
      Card card = c_get_card(code);
      card.set_location(public_location);''')
needle='    out["chain_size"] = pd->game_field->core.current_chain.size();'
assert t.count(needle)==1
t=t.replace(needle,needle+'''
    py::list chain;
    for(const auto& item:public_chain_) {
      py::dict row;row["code"]=item[0];row["player"]=item[1];row["description"]=item[2];chain.append(row);
    }
    out["public_chain"]=chain;
''')
h.write_text(t)
(R/'CMakeLists.txt').write_text((old/'CMakeLists.txt').read_text())
with (B/'build.log').open('w') as log:
    for command in [['cmake','-S',str(R),'-B',str(R/'build'),'-DCMAKE_BUILD_TYPE=Release','-DPython3_EXECUTABLE=/opt/ygo-ai/.venv/bin/python'],
                    ['cmake','--build',str(R/'build'),'-j','2']]:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Build failed; inspect '+str(B/'build.log'))
native=next((R/'agent/ygoenv/ygoenv/ygopro').glob('*.so'))
(B/'runtime.json').write_text(json.dumps(dict(runtime=str(R),native=str(native),sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            baseline_runtime=str(old),public_chain_source='public MSG_CHAINING packets; clear at reset and MSG_CHAIN_END',
            observation_shape_unchanged=True,extra_info_fields=['public_chain_length','public_chain_player','public_chain_card_id','public_chain_effect']),indent=2)+'\n')
print('BUILT_PUBLIC_CHAIN_RUNTIME',native)
