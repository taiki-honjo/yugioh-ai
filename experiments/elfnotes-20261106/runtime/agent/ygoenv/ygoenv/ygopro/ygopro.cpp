#include "ygoenv/ygopro/ygopro.h"
#include "ygoenv/core/py_envpool.h"

using YGOProEnvSpec = PyEnvSpec<ygopro::YGOProEnvSpec>;
using YGOProEnvPool = PyEnvPool<ygopro::YGOProEnvPool>;

#include "effect_checks.h"

PYBIND11_MODULE(ygopro_ygoenv, m) {
  m.def("run_effect_checks", &run_effect_checks);
  m.def("get_env_resource_stats", []() {
    pybind11::dict result;
    result["live_duels"] = ygopro::env_duels_live.load();
    result["peak_duels"] = ygopro::env_duels_peak.load();
    return result;
  });
  m.def("get_sum_search_stats", []() {
    pybind11::dict result;
    result["calls"] = ygopro::sum_search_calls.load();
    result["peak_candidates"] = ygopro::sum_search_peak_candidates.load();
    result["peak_results"] = ygopro::sum_search_peak_results.load();
    return result;
  });
  REGISTER(m, YGOProEnvSpec, YGOProEnvPool)

  m.def("init_module", &ygopro::init_module);
  m.def("get_lua_error_count", []() { return ygopro::lua_error_count.load(); });
}
