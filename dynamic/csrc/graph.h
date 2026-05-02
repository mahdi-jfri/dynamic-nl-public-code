#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>
#include <unordered_map>

namespace nlppr {

// Undirected graph with permanent self-loops. adj[i] contains all neighbors
// of i, including i itself exactly once (the self-loop). deg[i] = adj[i].size()
// is therefore always >= 1.
//
// pos[i] maps neighbor -> index in adj[i], so hasEdge / insertEdge / deleteEdge
// are O(1) average. adj is kept as a contiguous vector so the cleanup hot loop
// in cleanup_dim still iterates neighbors with cache-friendly stride.
class Graph {
public:
    int64_t n;
    std::vector<std::vector<int32_t>> adj;
    std::vector<std::unordered_map<int32_t, int32_t>> pos;
    std::vector<int32_t> deg;

    Graph() : n(0) {}

    void reset(int64_t nn) {
        n = nn;
        adj.assign((size_t)nn, {});
        pos.assign((size_t)nn, {});
        deg.assign((size_t)nn, 0);
        for (int64_t i = 0; i < nn; ++i) {
            adj[(size_t)i].push_back((int32_t)i);
            pos[(size_t)i].emplace((int32_t)i, 0);
            deg[(size_t)i] = 1;
        }
    }

    bool hasEdge(int32_t u, int32_t v) const {
        return pos[(size_t)u].count(v) > 0;
    }

    bool insertEdge(int32_t u, int32_t v) {
        auto& Pu = pos[(size_t)u];
        auto& Au = adj[(size_t)u];
        auto [it_u, inserted_u] = Pu.emplace(v, (int32_t)Au.size());
        if (!inserted_u) return false;
        Au.push_back(v);
        deg[(size_t)u]++;
        if (u != v) {
            auto& Pv = pos[(size_t)v];
            auto& Av = adj[(size_t)v];
            Pv.emplace(u, (int32_t)Av.size());
            Av.push_back(u);
            deg[(size_t)v]++;
        }
        return true;
    }

    // Refuses to delete self-loops (they are permanent).
    bool deleteEdge(int32_t u, int32_t v) {
        if (u == v) return false;
        auto remove_one = [this](int32_t a, int32_t b) -> bool {
            auto& Pa = pos[(size_t)a];
            auto it = Pa.find(b);
            if (it == Pa.end()) return false;
            int32_t idx = it->second;
            auto& Aa = adj[(size_t)a];
            int32_t last = Aa.back();
            if ((size_t)idx != Aa.size() - 1) {
                Aa[(size_t)idx] = last;
                Pa[last] = idx;
            }
            Aa.pop_back();
            Pa.erase(it);
            deg[(size_t)a]--;
            return true;
        };
        if (!remove_one(u, v)) return false;
        remove_one(v, u);
        return true;
    }
};

} // namespace nlppr
