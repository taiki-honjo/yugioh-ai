"""Paired base/head self-play with the verified public-information adapters.

Placement/position choices are externally controlled. Their rollout entries are
excluded from the PPO loss; the adapter's executed actions are not mislabeled as
samples from the neural policy. No teacher goals enter self-play observations.
"""
from pathlib import Path
import importlib.util
import json
import hashlib
import os
import shutil
import sys

control = None  # Bound by continuous_learning.backend() for supervisor operations.


def read(path):
    return json.loads(Path(path).read_text())


def policy_common(cfg):
    path = Path(cfg['root']) / 'snapshot/policy-code'
    sys.path.insert(0, str(path))
    import all_openings_common as common
    return common


def pin_helper_metadata(cfg,snapshot):
    C=control
    meta_run=Path(cfg['vocabulary']).parent
    C.atomic_copy(meta_run/'manifest.json',snapshot/'meta-run-manifest.json')
    path=snapshot/'policy-code/verify_opponent_article_basics.py'
    text=path.read_text()
    old="run=Path((P/'experiments/meta-opponents-20261107/latest-run.txt').read_text().strip())"
    if old in text:
        text=text.replace(old,f'run=Path({str(meta_run)!r})')
        text=text.replace("meta=json.loads((run/'manifest.json').read_text())",
                          f"meta=json.loads(Path({str(snapshot/'meta-run-manifest.json')!r}).read_text())")
        path.write_text(text)


def create_policy(cfg,checkpoint=None):
    common=policy_common(cfg)
    from center_placement import PlacementPolicy
    from battle_position import PositionPolicy
    checkpoint=Path(checkpoint or read(Path(cfg['root'])/'state.json')['best_checkpoint'])
    codes=common.v.v.engine.ID_CODES
    return PositionPolicy(PlacementPolicy(common.Policy(checkpoint/'base.flax_model',checkpoint/'chain-head.flax_model'),codes),codes)


def pack_bundle(directory, destination):
    import flax.serialization as serialization
    d = Path(directory)
    base = serialization.msgpack_restore((d / 'base.flax_model').read_bytes())
    head = serialization.msgpack_restore((d / 'chain-head.flax_model').read_bytes())
    value = {'params': {'base': base['params'], 'head': head['params']},
             'batch_stats': base.get('batch_stats', {})}
    Path(destination).write_bytes(serialization.to_bytes(value))


def unpack_bundle(combined, destination, source):
    import flax.serialization as serialization
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    value = serialization.msgpack_restore(Path(combined).read_bytes())
    original = serialization.msgpack_restore((Path(source) / 'base.flax_model').read_bytes())
    base = {'params': value['params']['base']}
    if 'batch_stats' in original:
        base['batch_stats'] = value['batch_stats']
    for name, data in [('base.flax_model', base), ('chain-head.flax_model', {'params': value['params']['head']})]:
        (destination / name).write_bytes(serialization.to_bytes(data))
    for name in ['metadata.json', 'runtime.json']:
        if (Path(source) / name).is_file():
            shutil.copyfile(Path(source) / name, destination / name)
    metadata=read(destination/'metadata.json') if (destination/'metadata.json').exists() else {}
    metadata.update(base_sha256=hashlib.sha256((destination/'base.flax_model').read_bytes()).hexdigest(),
                    head_sha256=hashlib.sha256((destination/'chain-head.flax_model').read_bytes()).hexdigest(),
                    source_bundle=str(source),combined_checkpoint=str(combined))
    (destination/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return destination


class CompositeAgent:
    def __init__(self, base_agent, head):
        self.base_agent, self.head = base_agent, head

    def init_rnn_state(self, *args, **kwargs):
        return self.base_agent.init_rnn_state(*args, **kwargs)

    def init(self, key, obs, state):
        import jax.numpy as jnp
        base = self.base_agent.init(key, self.base_obs(obs), state)
        n = obs['actions_'].shape[0]
        head = self.head.init(key, jnp.zeros((n, 4), jnp.int32), obs['actions_'])
        return {'params': {'base': base['params'], 'head': head['params']},
                'batch_stats': base.get('batch_stats', {})}

    @staticmethod
    def base_obs(obs):
        return {k: v for k, v in obs.items() if k not in ['chain_', 'adapter_controlled_']}

    def apply(self, variables, obs, *args, **kwargs):
        import jax.numpy as jnp
        base = {'params': variables['params']['base'], 'batch_stats': variables.get('batch_stats', {})}
        result = self.base_agent.apply(base, self.base_obs(obs), *args, **kwargs)
        mutable = kwargs.get('mutable', False)
        outputs, updates = result if mutable else (result, None)
        chain = obs.get('chain_', jnp.zeros((obs['actions_'].shape[0], 4), jnp.int32))
        delta = self.head.apply({'params': variables['params']['head']}, chain, obs['actions_'])
        outputs = (outputs[0], outputs[1] + delta, *outputs[2:])
        return (outputs, updates) if mutable else outputs


def augmented_preprocess(original, features, owner=None):
    import numpy as np
    class ChainObservations(original):
        def augment(self, obs, info):
            obs['chain_'] = features(info)
            obs['adapter_controlled_'] = np.isin(np.asarray(obs['actions_'])[:, 0, 3], [5, 11])
            if owner is not None:
                owner._opening_infos = info
            return obs

        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            if owner is not None:
                for bank in getattr(owner,'_opening_banks',{}).values():
                    for policy in bank.controllers.values():policy.reset()
            return self.augment(obs, info), info

        def step(self, actions):
            obs, reward, term, trunc, info = super().step(actions)
            return self.augment(obs, info), reward, term, trunc, info
    return ChainObservations


class AdapterBank:
    """Separate public-action memory for each environment and actual player."""
    def __init__(self, cfg, n):
        import numpy as np
        common = policy_common(cfg)
        from center_placement import PlacementPolicy
        from battle_position import PositionPolicy
        codes = {i + 1: int(line.split()[0]) for i, line in enumerate(Path(cfg['vocabulary']).read_text().splitlines())}
        class Scores:
            def __init__(self): self.base = self.head = self.state = None; self.logits = None
            def reset(self): self.state = None
            def infer(self, *args): return None, self.logits
        self.controllers = {}
        self.scores = {}
        for slot in range(n):
            for seat in [0, 1]:
                scores = Scores()
                self.scores[slot, seat] = scores
                self.controllers[slot, seat] = PositionPolicy(PlacementPolicy(scores, codes), codes)
        self.changed = 0
        self.controlled = 0

    def apply(self, obs, info, raw, logits, done, active=None):
        import numpy as np
        raw = np.asarray(raw).copy()
        result = raw.copy()
        active = np.ones(len(raw), bool) if active is None else np.asarray(active)
        for slot in range(len(raw)):
            if not active[slot]: continue
            if done[slot]:
                for seat in [0, 1]: self.controllers[slot, seat].reset()
            seat = int(info['to_play'][slot])
            controller = self.controllers[slot, seat]
            scores = np.asarray(logits)[slot:slot+1].copy()
            # Preserve the sampled action as the adapter input, keeping the order
            # of all alternative legal choices. No PPO gradient is used at either
            # externally controlled message type (5:position / 11:place).
            scores[0, int(raw[slot])] = np.max(scores) + 1.
            self.scores[slot, seat].logits = scores
            one_obs = {k: None if v is None else np.asarray(v)[slot:slot+1] for k, v in obs.items()}
            one_info = {k: np.asarray(v)[slot:slot+1] for k, v in info.items() if hasattr(v, '__len__') and len(v) == len(raw)}
            result[slot] = controller.act(one_obs, one_info)
            n = int(info['num_options'][slot])
            if not 0 <= result[slot] < n: raise RuntimeError('Adapter selected an illegal action')
            controlled = int(np.asarray(obs['actions_'])[slot, int(raw[slot]), 3]) in [5, 11]
            if result[slot] != raw[slot] and not controlled:
                raise RuntimeError('Adapter modified an unmasked neural-policy action')
            self.controlled += int(controlled)
            self.changed += int(result[slot] != raw[slot])
        return result


def install_training_backend(namespace):
    cfg = read(os.environ['YGO_CONTINUOUS_CONFIG'])
    common = policy_common(cfg)
    original_agent = namespace['create_agent']
    namespace['create_agent'] = lambda args, eval=False: CompositeAgent(original_agent(args, eval), common.HEAD)
    namespace['EnvPreprocess'] = augmented_preprocess(namespace['EnvPreprocess'], common.features)
    namespace['OpeningAdapterBank'] = lambda n: AdapterBank(cfg, n)


def patched_trainer(text):
    patches = [
        ('    main()\n', '    from continuous_opening_backend import install_training_backend\n    install_training_backend(globals())\n    main()\n'),
        ('\n    next_to_play = info["to_play"]\n', '\n    opening_adapters = OpeningAdapterBank(args.local_num_envs)\n    next_to_play = info["to_play"]\n'),
        ('            cpu_action = np.array(action)\n', '            cpu_action = opening_adapters.apply(cached_next_obs, info, np.array(action), np.array(logits), np.array(cached_next_done))\n'),
        ('        n_valids = jnp.sum(mask)\n', '        n_valids = jnp.maximum(jnp.sum(mask), 1)\n'),
        ('            device_params["params"]["Encoder_0"][\'Embed_0\']["embedding"].block_until_ready()\n',
         '            jax.block_until_ready(device_params)\n'),
    ]
    for old, new in patches:
        if text.count(old) != 1: raise RuntimeError(f'Unexpected trainer source: {old!r}')
        text = text.replace(old, new)
    old = '        loss, pg_loss, v_loss, ent_loss, approx_kl = compute_loss(\n'
    if text.count(old) != 2: raise RuntimeError('Expected both PPO loss call sites')
    text = text.replace(old, "        mask = jnp.logical_and(mask, ~jnp.reshape(obs['adapter_controlled_'], mask.shape))\n" + old)
    compile(text, 'integrated-trainer.py', 'exec')
    return text


def initialize(cfg):
    C = control; root = Path(cfg['root']); root.mkdir(parents=True, exist_ok=True)
    if (root / 'state.json').exists():
        C.runtime_check(cfg)
        return read(root / 'state.json')
    source_policy = read(cfg['verified_policy_config'])
    source = Path(source_policy['selected_checkpoint'])
    for name, key in [('base.flax_model','base_sha256'),('chain-head.flax_model','head_sha256')]:
        if C.sha(source / name) != source_policy[key]: raise RuntimeError('Verified model checksum mismatch')
    snapshot = root / 'snapshot'; snapshot.mkdir(exist_ok=True)
    code = Path(source_policy['placement_adapter']['runtime_code_directory'])
    (snapshot / 'policy-code').mkdir(exist_ok=True)
    for path in code.iterdir():
        if path.is_file() and path.suffix in ['.py','.json']:
            C.atomic_copy(path, snapshot / 'policy-code' / path.name)
    pin_helper_metadata(cfg,snapshot)
    C.atomic_copy(cfg['verified_policy_config'], snapshot / 'opening-policy.json')
    C.atomic_copy(cfg['opening_manifest'], snapshot / 'opening-manifest.json')
    C.atomic_copy(cfg['runtime_manifest'], snapshot / 'runtime.json')
    C.atomic_copy(Path(cfg['project']) / 'scripts/evaluate.py', snapshot / 'evaluator.py')
    C.atomic_copy(Path(__file__), snapshot / 'continuous_opening_backend.py')
    text = patched_trainer((Path(cfg['runtime']) / 'agent/scripts/cleanba.py').read_text())
    (snapshot / 'train.py').write_text(text)
    C.atomic_json(snapshot / 'hashes.json', {str(p.relative_to(snapshot)): C.sha(p) for p in snapshot.rglob('*') if p.is_file() and p.name!='hashes.json'})
    initial = root / 'models/initial'; initial.mkdir(parents=True, exist_ok=True)
    for name in ['base.flax_model','chain-head.flax_model']: C.atomic_copy(source / name, initial / name)
    C.atomic_json(initial / 'metadata.json', dict(source_checkpoint=str(source), base_sha256=C.sha(initial/'base.flax_model'),
                  head_sha256=C.sha(initial/'chain-head.flax_model'), adapters_enabled=True, normal_summon_usage_scored=False))
    C.atomic_copy(snapshot / 'runtime.json', initial / 'runtime.json')
    (root / 'cycles').mkdir(exist_ok=True)
    state = dict(schema_version=2, backend='verified_opening', created_jst=C.now(), status='stopped', phase='idle',
                 completed_cycles=0, cycle=0, added_steps=0, best_added_steps=0, failures=0, pending=None,
                 latest_checkpoint=str(initial), best_checkpoint=str(initial), initial_checkpoint=str(source),
                 initial_sha256=C.sha(initial/'base.flax_model'), initial_head_sha256=C.sha(initial/'chain-head.flax_model'),
                 pid=None,last_error=None,network_or_llm_calls=False,
                 resume_kind='base and chain-head weights; optimizer/RNG/duel state restart each chunk',
                 regression_policy='all initially passing validation openings must remain passing; holdout is not used')
    C.save_state(cfg,state); C.runtime_check(cfg)
    return state


def train_cycle(cfg, state, pending):
    C=control; directory=Path(pending['directory']); checkpoints=directory/'checkpoints'; checkpoints.mkdir()
    packed=directory/'source-combined.flax_model'; pack_bundle(pending['source'],packed)
    steps=pending['steps'];updates=steps//128
    command=[cfg['python'],'-B','-u',str(Path(cfg['root'])/'snapshot/train.py'),
             '--actor-device-ids','0','--learner-device-ids','0','--deck',cfg['deck'],
             '--code-list-file',cfg['vocabulary'],'--max-options','128','--timeout','30',
             '--local-num-envs','4','--local-env-threads','2','--num-actor-threads','1',
             '--num-steps','32','--num-minibatches','4','--learning-rate',str(cfg['learning_rate']),
             '--vloss-clip','1.0','--save-interval',str(min(32,updates)),
             '--local-eval-episodes','4','--eval-interval',str(updates),'--total-timesteps',str(steps),
             '--seed',str(pending['training_seed']),'--log-frequency','8',
             '--ckpt-dir',str(checkpoints),'--tb-dir',str(directory/'tensorboard'),'--checkpoint',str(packed)]
    log_path=C.run_job(cfg,state,command,directory,'train')
    return finalize_training(cfg,pending,log_path)


def finalize_training(cfg,pending,log_path):
    C=control;directory=Path(pending['directory']);checkpoints=directory/'checkpoints'
    packed=directory/'source-combined.flax_model';steps=pending['steps']
    log=Path(log_path).read_text(errors='replace')
    if 'NATIVE_LUA_ERRORS 0' not in log:
        raise RuntimeError('Integrated training completion check failed')
    paths=list(checkpoints.glob(f'*_{steps}steps.flax_model'))
    if len(paths)!=1:raise RuntimeError('Expected a complete combined checkpoint')
    result=C.validate_weights(packed,checkpoints/'initial.flax_model',paths[0])
    candidate=unpack_bundle(paths[0],directory/'candidate',pending['source'])
    result.update(checkpoint=str(candidate),steps=steps,base_sha256=C.sha(candidate/'base.flax_model'),
                  head_sha256=C.sha(candidate/'chain-head.flax_model'),both_networks_optimized=True,
                  placement_and_position_adapters_used_in_rollout=True,adapter_steps_excluded_from_policy_loss=True)
    C.atomic_json(directory/'training-result.json',result)
    return candidate


def openings_job(args,cfg):
    common=policy_common(cfg)
    import all_opening_cases as cases
    manifest=read(Path(cfg['root'])/'snapshot/opening-manifest.json')
    checkpoint=Path(args.candidate)
    policy=create_policy(cfg,checkpoint)
    rows=[];seen=set()
    for case in manifest['decks']['ElfnotesKewlTune']['cases']['validation']:
        key=(case['eval_family'],tuple(sorted(case['hand'])),case['seed'])
        if key in seen:continue
        seen.add(key)
        result,_=common.v.run(cases.normalize(case),policy=policy)
        rows.append(dict(case_id=case['case_id'],family=case['eval_family'],passed=bool(result['passed']),
                         lua_errors=result['lua_errors'],steps=len(result['trace'])))
        print('OPENING',rows[-1],flush=True)
    passed=[r['case_id'] for r in rows if r['passed']]
    control.atomic_json(args.result,dict(status='complete',split='validation',total=len(rows),passed=len(passed),
                        passed_case_ids=passed,cases=rows,lua_errors=common.v.native.get_lua_error_count(),
                        model_received_goals_or_plans=False,normal_summon_usage_scored=False,
                        checkpoint=str(checkpoint),adapters_enabled=True,holdout_used=False))


def regression_ok(baseline,candidate):
    return (candidate.get('status')=='complete' and candidate.get('total')==baseline['total']
            and candidate.get('lua_errors')==0 and candidate.get('normal_summon_usage_scored') is False
            and not candidate.get('holdout_used',True)
            and set(baseline['passed_case_ids']).issubset(candidate.get('passed_case_ids',[])))


def regression_gate(cfg,state,pending,candidate):
    C=control;root=Path(cfg['root']);directory=Path(pending['directory'])
    baseline_path=root/'opening-baseline.json'
    def run(path,checkpoint,name):
        C.run_job(cfg,state,[cfg['python'],'-B','-u',str(Path(cfg['project'])/'scripts/continuous_learning.py'),
                   'job-openings','--config',cfg['_file'],'--candidate',str(checkpoint),'--result',str(path)],directory,name)
        return read(path)
    if not baseline_path.exists():
        baseline=run(baseline_path,root/'models/initial','opening-baseline')
        if baseline['total']!=66 or baseline['lua_errors']!=0:raise RuntimeError('Opening baseline is incomplete')
    else:baseline=read(baseline_path)
    if (root/'opening-best.json').exists():baseline=read(root/'opening-best.json')
    path=directory/'opening-regression.json'
    current=run(path,candidate,'opening-regression')
    return dict(passed=regression_ok(baseline,current),baseline_passed=baseline['passed'],
                candidate_passed=current['passed'],total=current['total'],evidence=str(path),
                lost_case_ids=sorted(set(baseline['passed_case_ids'])-set(current['passed_case_ids'])))


def recover(cfg,state):
    C=control;pending=state.get('pending')
    if not pending or pending['phase']!='training':return
    directory=Path(pending['directory'])
    import re
    paths=list((directory/'checkpoints').glob('*steps.flax_model'))
    paths.sort(key=lambda p:int(re.search(r'_(\d+)steps',p.name)[1]),reverse=True)
    for path in paths:
        try:
            C.validate_weights(directory/'source-combined.flax_model',directory/'checkpoints/initial.flax_model',path)
            bundle=unpack_bundle(path,directory/'recovered',pending['source'])
        except Exception:continue
        steps=int(re.search(r'_(\d+)steps',path.name)[1])
        state.update(latest_checkpoint=str(bundle),added_steps=pending['starting_added_steps']+steps)
        C.event(cfg,'recovered_integrated_checkpoint',checkpoint=str(bundle),steps=steps)
        break
    state['pending']=None;C.save_state(cfg,state)


def prune(cfg,state):
    root=Path(cfg['root']).resolve()
    protected=[Path(state[k]).resolve() for k in ['best_checkpoint','latest_checkpoint']]
    if state.get('pending'):
        protected.extend(Path(state['pending'][k]).resolve() for k in ['source','directory'])
    directories=sorted((root/'cycles').iterdir(),key=lambda p:p.name)
    for directory in directories[:-cfg['keep_cycle_checkpoints']]:
        actual=directory.resolve()
        if actual.parent!=root/'cycles' or directory.is_symlink():raise RuntimeError('Unsafe retention path')
        if any(p.is_relative_to(actual) for p in protected):continue
        for path in directory.rglob('*.flax_model'):
            if not path.is_symlink() and path.resolve().is_relative_to(actual):path.unlink()


def install_evaluation(ctx,cfg):
    import jax
    import jax.numpy as jnp
    import numpy as np
    common=policy_common(cfg)
    agent=CompositeAgent(ctx.agent,common.HEAD);ctx.agent=agent
    ctx.EnvPreprocess=augmented_preprocess(ctx.EnvPreprocess,common.features,ctx)
    cache={};banks={};ctx._opening_banks=banks
    def load(spec):
        checkpoint=Path(spec['path']).parent
        key=str(checkpoint)
        if key not in cache:
            import flax.serialization as ser
            base=ser.msgpack_restore((checkpoint/'base.flax_model').read_bytes())
            head=ser.msgpack_restore((checkpoint/'chain-head.flax_model').read_bytes())
            cache[key]=jax.device_put({'params':dict(base=base['params'],head=head['params']),
                                      'batch_stats':base.get('batch_stats',{})})
            banks[id(cache[key])]=AdapterBank(cfg,ctx.num_envs)
        return cache[key]
    @jax.jit
    def infer(params,state,obs):
        result=agent.apply(params,obs,state)
        return result[0],result[1]
    def step(params,state,obs,active,done):
        state=jax.tree.map(lambda s:jnp.where(done[:,None],0,s),state)
        next_state,logits=infer(params,state,obs)
        state=jax.tree.map(lambda n,o:jnp.where(active[:,None],n,o),next_state,state)
        scores=np.asarray(logits);actions=scores.argmax(-1)
        actions=banks[id(params)].apply(obs,ctx._opening_infos,actions,scores,np.asarray(done),np.asarray(active))
        return state,actions
    ctx.load_params=load;ctx._step_model=step
    # Initial compilation is included in this backend's batch wall time. Scores
    # are comparable; these timing fields are not a pure inference benchmark.
    ctx.warmup_jit=lambda:None
