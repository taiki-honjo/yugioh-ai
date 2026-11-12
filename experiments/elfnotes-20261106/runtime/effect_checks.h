
#pragma once
#include "ygopro-core/field.h"
#include "ygopro-core/card.h"
#include "ygopro-core/effect.h"
#include "ygopro-core/interpreter.h"

inline pybind11::list run_effect_checks() {
  pybind11::list results;
  auto record = [&](const char* name, bool actual, bool expected) {
    pybind11::dict row;
    row["name"] = name; row["actual"] = actual; row["expected"] = expected;
    row["pass"] = actual == expected; results.append(row);
  };
  auto fresh = []() {
    auto handle = create_duel(702);
    set_player_info(handle, 0, 8000, 0, 0);
    set_player_info(handle, 1, 8000, 0, 0);
    auto* pd = reinterpret_cast<duel*>(handle);
    pd->game_field->infos.turn_player = 0;
    pd->game_field->infos.phase = PHASE_MAIN1;
    pd->game_field->core.duel_rule = 5;
    return handle;
  };
  for (int zone : {0, 2, 4}) {
    auto handle = fresh();
    new_card(handle, 56651978, 0, 0, LOCATION_MZONE, zone, POS_FACEUP_ATTACK);
    auto* pd = reinterpret_cast<duel*>(handle);
    auto* regina = pd->game_field->player[0].list_mzone[zone];
    bool found = false, condition = false;
    auto range = regina->single_effect.equal_range(EVENT_SPSUMMON_SUCCESS);
    for (auto it=range.first; it!=range.second; ++it) {
      auto* e = it->second;
      pd->lua->add_param(e, PARAM_TYPE_EFFECT);
      condition = pd->lua->check_condition(e->condition, 1) != 0;
      found = true;
    }
    std::string name = "regina_recruit_condition_zone_" + std::to_string(zone);
    record(name.c_str(), found && condition, zone == 2);
    end_duel(handle);
  }
  for (uint32_t hand : {0u, 23434538u, 16509007u}) {
    auto handle = fresh();
    new_card(handle, 43904702, 0, 0, LOCATION_MZONE, 0, POS_FACEUP_ATTACK);
    if (hand) new_card(handle, hand, 0, 0, LOCATION_HAND, 0, POS_FACEDOWN_DEFENSE);
    new_card(handle, 42781164, 0, 0, LOCATION_EXTRA, 0, POS_FACEDOWN_DEFENSE);
    auto* pd = reinterpret_cast<duel*>(handle);
    auto* target = pd->game_field->player[0].list_extra.back();
    bool summonable = target->is_special_summonable(0, SUMMON_TYPE_SYNCHRO) != 0;
    std::string name = "clip_hand_synchro_hand_" + std::to_string(hand);
    record(name.c_str(), summonable, hand == 16509007u);
    end_duel(handle);
  }
  return results;
}
