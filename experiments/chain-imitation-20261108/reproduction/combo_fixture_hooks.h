  pybind11::dict benchmark_snapshot() {
    namespace py = pybind11;
    py::dict out;
    out["to_play"] = int(to_play_); out["message"] = msg_to_string(msg_);
    out["message_id"] = msg_; out["done"] = done_; out["steps"] = step_count_;
    auto* pd = reinterpret_cast<duel*>(pduel_);
    out["turn_player"] = pd->game_field->infos.turn_player;
    out["turn_count"] = turn_count_; out["phase"] = pd->game_field->infos.phase;
    out["chain_size"] = pd->game_field->core.current_chain.size();
    out["normal_summons"] = pd->game_field->core.summon_count[0];
    out["lp0"] = lp_[0]; out["lp1"] = lp_[1];
    py::list all;
    auto append = [&](card* c) {
      if (!c) return;
      py::dict row;
      row["code"] = c->data.code; row["controller"] = int(c->current.controler);
      row["location"] = int(c->current.location); row["sequence"] = int(c->current.sequence);
      row["position"] = int(c->current.position); row["level"] = c->get_level();
      all.append(row);
    };
    for (int p=0;p<2;++p) {
      auto& state = pd->game_field->player[p];
      for(auto* c: state.list_hand) append(c);
      for(auto* c: state.list_mzone) append(c);
      for(auto* c: state.list_szone) append(c);
      for(auto* c: state.list_grave) append(c);
      for(auto* c: state.list_remove) append(c);
      for(auto* c: state.list_extra) append(c);
    }
    out["cards"] = all;
    py::list actions;
    for (int i=0;i<int(legal_actions_.size());++i) {
      const auto& a=legal_actions_[i];
      py::dict row;
      row["index"]=i; row["act"]=action_act_to_string(a.act_);
      row["phase"]=action_phase_to_string(a.phase_); row["spec"]=a.spec_;
      row["place"]=int(a.place_); row["finish"]=a.finish_;
      row["position"]=int(a.position_); row["number"]=int(a.number_);
      row["effect"]=a.effect_; row["response"]=a.response_;
      CardId cid=a.cid_;
      if(!cid && !a.spec_.empty()) {
        try {cid=spec_to_card_id(a.spec_,to_play_);} catch(const std::exception&) {}
      }
      CardCode code=0;
      for(const auto& entry:card_ids_) if(entry.second==cid) {code=entry.first;break;}
      row["code"]=code;
      actions.append(row);
    }
    out["actions"]=actions;
    return out;
  }
