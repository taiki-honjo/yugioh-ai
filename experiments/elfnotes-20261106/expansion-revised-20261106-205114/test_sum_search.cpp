
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>
inline std::vector<std::vector<int>> combinations(int n, int r) {
  std::vector<std::vector<int>> combs;
  std::vector<bool> m(n);
  std::fill(m.begin(), m.begin() + r, true);

  do {
    std::vector<int> cs;
    cs.reserve(r);
    for (int i = 0; i < n; ++i) {
      if (m[i]) {
        cs.push_back(i);
      }
    }
    combs.push_back(cs);
  } while (std::prev_permutation(m.begin(), m.end()));

  return combs;
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind, int i,
                   int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  if (i == n - 1) {
    return r == 1 || (w[ind[i]] == r);
  }
  return sum_to(w, ind, i + 1, r - 1) || sum_to(w, ind, i + 1, r - w[ind[i]]);
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind,
                   int r) {
  return sum_to(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight(const std::vector<int> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int i, int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  const auto &w_ = w[ind[i]];
  if (i == n - 1) {
    if (w_.size() == 1) {
      return w_[0] == r;
    } else {
      return w_[0] == r || w_[1] == r;
    }
  }
  if (w_.size() == 1) {
    return sum_to2(w, ind, i + 1, r - w_[0]);
  } else {
    return sum_to2(w, ind, i + 1, r - w_[0]) ||
           sum_to2(w, ind, i + 1, r - w_[1]);
  }
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int r) {
  return sum_to2(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
reference_combinations_with_weight2(
  const std::vector<std::vector<int>> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to2(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline std::vector<std::vector<int>>
combinations_with_weight2(const std::vector<std::vector<int>> &weights, int target) {
  // Preserve cardinality/lexicographic order and every valid subset. Avoid
  // materializing all 2^N subsets merely to reject almost all of them by sum.
  const int n = static_cast<int>(weights.size());
  std::vector<std::vector<int>> results;
  if (target <= 0 || n == 0) return results;
  for (const auto& w : weights) {
    if (w.empty()) throw std::runtime_error("Empty synchro weight");
  }
  std::map<std::tuple<int,int,int>, bool> memo;
  std::function<bool(int,int,int)> reachable = [&](int begin, int count, int remaining) {
    if (count == 0) return remaining == 0;
    if (remaining <= 0 || n-begin < count) return false;
    auto key = std::make_tuple(begin,count,remaining);
    auto found = memo.find(key);
    if (found != memo.end()) return found->second;
    bool possible = reachable(begin+1,count,remaining);
    if (!possible) {
      const auto& w = weights[begin];
      const int choices = w.size() == 1 ? 1 : 2;
      for (int j=0; j<choices; ++j) {
        const int next = remaining-w[j];
        if ((count == 1 && next == 0) ||
            (count > 1 && reachable(begin+1,count-1,next))) {
          possible=true; break;
        }
      }
    }
    memo.emplace(key,possible);
    return possible;
  };
  std::vector<int> path;
  std::function<void(int,int,const std::vector<int>&)> enumerate =
      [&](int begin,int count,const std::vector<int>& remaining) {
    if (count == 0) { results.push_back(path); return; }
    for (int i=begin; i<=n-count; ++i) {
      std::vector<int> next;
      const auto& w=weights[i];
      const int choices = w.size() == 1 ? 1 : 2;
      for (int total : remaining) {
        if (total <= 0) continue;
        for (int j=0;j<choices;++j) {
          int value=total-w[j];
          if ((count==1 && value==0) || (count>1 && reachable(i+1,count-1,value)))
            next.push_back(value);
        }
      }
      std::sort(next.begin(),next.end());
      next.erase(std::unique(next.begin(),next.end()),next.end());
      if (next.empty()) continue;
      path.push_back(i);
      enumerate(i+1,count-1,next);
      path.pop_back();
    }
  };
  for (int count=1;count<=n;++count) {
    if (reachable(0,count,target)) enumerate(0,count,std::vector<int>{target});
  }
  return results;
}

int main() {
  std::mt19937 rng(70326);
  for (int t=0;t<3000;++t) {
    int n=rng()%11;
    std::vector<std::vector<int>> weights;
    for (int i=0;i<n;++i) {
      std::vector<int> w{int(rng()%8)-1};
      if(rng()%2) w.push_back(int(rng()%8)-1);
      weights.push_back(w);
    }
    int target=int(rng()%18)-2;
    auto a=reference_combinations_with_weight2(weights,target);
    auto b=combinations_with_weight2(weights,target);
    if(a!=b) { std::cerr<<"MISMATCH "<<t<<"\n"; return 1; }
  }
  auto start=std::chrono::steady_clock::now();
  auto stress=combinations_with_weight2(std::vector<std::vector<int>>(40,{1}),2);
  if(stress.size()!=780) return 2;
  auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::cout<<"{\"parity_cases\":3000,\"stress_candidates\":40,\"stress_result_count\":780,\"stress_ms\":"<<ms<<"}\n";
}
