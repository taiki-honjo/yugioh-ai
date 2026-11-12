"""Opponent scripts, common fallback and stop rule for the two-landing comparison (pure, no engine).

Opponent = player 1, turn 2.
  search   O1: Rukina (1) special summon, Rukina (2) search (target Spirit when offered).
  synchro  the teacher route of the TRAIN case elf-rukina-regina-g00-p0 (Rukina -> Regina -> Strelitzia -> Crystal Wing
           -> Baronne ...), written for player 0 and driven here for player 1 through a mirrored snapshot (controllers and
           chain players swapped; the legal-action list is untouched, so the returned index is valid as is).
When a script cannot continue (its next planned action is not offered at the idle command, or the teacher raises),
the SAME adopted model plays the rest of the opponent's turn 2 in every condition (common fallback). After a script
completes, the opponent ends its turn. During our turn 3 the opponent only passes (common rule)."""
import copy

RUK, SPIRIT = 13597785, 12375297


def mirror(raw):
    m = dict(raw)
    m['cards'] = [dict(c, controller=1 - c['controller']) for c in raw['cards']]
    m['public_chain'] = [dict(x, player=1 - x['player']) for x in raw['public_chain']]
    m['turn_player'], m['to_play'] = 1 - raw['turn_player'], 1 - raw['to_play']
    m['lp0'], m['lp1'] = raw.get('lp1'), raw.get('lp0')
    return m


class SearchScript:
    """O1 with explicit block detection (the original O1 helper would silently end the turn)."""
    def __init__(self):
        self.steps = [dict(act='SpSummon', code=RUK), dict(act='Activate', code=RUK)]

    def __call__(self, raw, passive):
        acts, msg = raw['actions'], raw['message']
        if raw['phase'] == 4 and self.steps:
            target = self.steps[0]
            found = next((a for a in acts if all(a[k] == v for k, v in target.items())), None)
            if found:
                self.steps.pop(0)
                return found['index'], 'script'
            if msg == 'select_idlecmd':
                raise LookupError(f'planned {target} not offered')
        if raw['phase'] == 4:
            if msg == 'select_place':
                return next((a['index'] for a in acts if a['place'] == 3), acts[0]['index']), 'script'
            if msg == 'select_position':
                return 0, 'script'
            if msg in ['select_card', 'select_unselect_card']:
                return next((a['index'] for a in acts if a['code'] == SPIRIT), 0), 'script'
        return passive(raw), 'script-pass'


class TeacherScript:
    def __init__(self, teacher):
        self.teacher = teacher

    @property
    def steps(self):
        return list(getattr(self.teacher, 'steps', []))

    def __call__(self, raw, passive):
        return self.teacher(mirror(raw)), 'script'


class Opponent:
    """script -> (on block) common fallback model for the rest of turn 2; passive in turn 3."""
    def __init__(self, script, fallback_act, passive):
        self.script, self.fallback_act, self.passive = script, fallback_act, passive
        self.mode, self.blocked, self.labels = 'script', None, []

    def decide(self, raw, obs, info, step):
        if raw['turn_count'] >= 3:
            return self._passive(raw), 'turn3-pass'
        if self.mode == 'script':
            try:
                return self.script(raw, self.passive)
            except Exception as exc:  # cannot continue: record and hand over to the common fallback
                self.mode = 'fallback'
                self.blocked = dict(step=step, message=raw['message'], phase=raw['phase'], error=f'{type(exc).__name__}: {exc}'[:800],
                                    next_plan_steps=[dict(x) for x in self.script.steps[:4]],
                                    legal=[(a.get('act'), a.get('code'), a.get('effect'), a.get('spec')) for a in raw['actions']])
        return self.fallback_act(obs, info), 'fallback-model'

    def _passive(self, raw):
        try:
            return self.passive(raw)
        except Exception:
            return raw['actions'][0]['index']


def finished(raw):
    """Stop when the opponent's turn 4 begins (our turn 3 is over)."""
    return raw['turn_count'] >= 4
