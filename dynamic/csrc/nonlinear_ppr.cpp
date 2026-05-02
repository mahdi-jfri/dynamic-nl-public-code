#include "nonlinear_ppr.h"
#include <cmath>
#include <cstring>
#include <queue>
#include <cstdint>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nlppr {

NonlinearPPR::NonlinearPPR(int64_t n_, int F_, double alpha_, double beta_,
                           double K_, int step_fn_code, double step_param_,
                           int threshold_mode_code)
    : n(n_), F(F_), alpha(alpha_), beta(beta_), K(K_),
      step_fn((StepFn)step_fn_code), step_param(step_param_),
      threshold_mode((ThresholdMode)threshold_mode_code),
      state_dirty(true) {
    g.reset(n);
    X.assign((size_t)F, std::vector<double>((size_t)n, 0.0));
    Z.assign((size_t)F, std::vector<double>((size_t)n, 0.0));
    Y.assign((size_t)F, std::vector<double>((size_t)n, 0.0));
    R.assign((size_t)F, std::vector<double>((size_t)n, 0.0));

    // All initial degrees are 1 (self-loops only), so d^(1-beta) = d^beta = 1.
    deg_1mb.assign((size_t)n, 1.0);
    deg_b.assign((size_t)n, 1.0);

    active_queue.assign((size_t)F, {});
    in_queue.assign((size_t)F, std::vector<uint8_t>((size_t)n, 0));
    update_w.reserve((size_t)F);
    affected_idx_buf.assign((size_t)n, (int32_t)-1);
    rowsum_pos.assign((size_t)F, 0.0);
    rowsum_neg.assign((size_t)F, 0.0);
}

void NonlinearPPR::refresh_degree_cache(int32_t i) {
    double di = (double)g.deg[(size_t)i];
    deg_1mb[(size_t)i] = std::pow(di, 1.0 - beta);
    deg_b[(size_t)i]   = std::pow(di, beta);
}

void NonlinearPPR::set_features(const double* X_in) {
    for (int d = 0; d < F; ++d) {
        std::memcpy(X[(size_t)d].data(), X_in + (size_t)d * (size_t)n,
                    sizeof(double) * (size_t)n);
        double p = 0.0, q = 0.0;
        const double* Xd = X[(size_t)d].data();
        for (size_t i = 0; i < (size_t)n; ++i) {
            if (Xd[i] > 0.0) p += Xd[i];
            else             q += Xd[i];
        }
        rowsum_pos[(size_t)d] = p;
        rowsum_neg[(size_t)d] = q;  // <= 0
    }
}

double NonlinearPPR::apply_step(double x) const {
    switch (step_fn) {
    case SF_IDENTITY: return x;
    case SF_SIGMOID:  return 1.0 / (1.0 + std::exp(-x));
    case SF_TANH:     return std::tanh(x);
    case SF_SOFTPLUS: {
        if (x > 30.0) return x;
        return std::log1p(std::exp(x));
    }
    case SF_RELU: return x > 0.0 ? x : 0.0;
    case SF_CLAMP_SYM: {
        double p = step_param > 0 ? step_param : 1.0;
        return x < -p ? -p : (x > p ? p : x);
    }
    case SF_STANH: {
        double c = step_param > 0 ? step_param : 1.0;
        return std::tanh(c * x) / c;
    }
    case SF_LEAKY05: return x >= 0.0 ? x : 0.5 * x;
    case SF_SHTANH:  return std::tanh(x - step_param);
    case SF_HTANH: {
        double w = step_param > 0 ? step_param : 1.0;
        return x < -w ? -w : (x > w ? w : x);
    }
    }
    return x;
}

double NonlinearPPR::threshold_i(int32_t i, double eps) const {
    double base = (1.0 - K * (1.0 - alpha)) * eps;
    double d = (double)g.deg[(size_t)i];
    return base * std::pow(d, 1.0 - beta);
}

void NonlinearPPR::reset_state() {
    for (int d = 0; d < F; ++d) {
        auto& Zd = Z[(size_t)d];
        auto& Yd = Y[(size_t)d];
        auto& Rd = R[(size_t)d];
        const auto& Xd = X[(size_t)d];
        std::memset(Zd.data(), 0, sizeof(double) * (size_t)n);
        for (size_t i = 0; i < (size_t)n; ++i) {
            Yd[i] = alpha * Xd[i];
            Rd[i] = apply_step(Yd[i]);  // z=0, so r = U(y) - 0
        }
        std::queue<int32_t>().swap(active_queue[(size_t)d]);
        std::fill(in_queue[(size_t)d].begin(), in_queue[(size_t)d].end(),
                  (uint8_t)0);
    }
    state_dirty = true;
}

double NonlinearPPR::initial_operation(const int32_t* edges, int64_t E,
                                       double eps) {
    for (int64_t i = 0; i < E; ++i) {
        g.insertEdge(edges[2 * i], edges[2 * i + 1]);
    }
    // One-shot O(n) refresh after bulk insertion; cheap relative to cleanup.
    for (int32_t i = 0; i < (int32_t)n; ++i) refresh_degree_cache(i);
    reset_state();
    return cleanup(eps);
}

double NonlinearPPR::cleanup_dim(int d, double eps) {
    double* Zd = Z[(size_t)d].data();
    double* Yd = Y[(size_t)d].data();
    double* Rd = R[(size_t)d].data();

    const ThresholdMode tm = threshold_mode;
    double base_thr = (1.0 - K * (1.0 - alpha)) * eps;
    double ub_d = (tm == TH_ROWSUM) ? alpha * rowsum_pos[(size_t)d] * eps : 0.0;
    double lb_d = (tm == TH_ROWSUM) ? alpha * rowsum_neg[(size_t)d] * eps : 0.0;
    double one_minus_alpha = 1.0 - alpha;

    auto& q = active_queue[(size_t)d];
    auto& inq = in_queue[(size_t)d];

    double work = 0.0;
    while (!q.empty()) {
        int32_t i = q.front();
        q.pop();
        inq[(size_t)i] = 0;

        double delta = Rd[i];
        Zd[i] += delta;

        double di_1mb_i = deg_1mb[(size_t)i];
        const auto& Ni = g.adj[(size_t)i];
        size_t deg_i = Ni.size();
        for (size_t a = 0; a < deg_i; ++a) {
            int32_t j = Ni[a];
            double w_ji = 1.0 / (deg_b[(size_t)j] * di_1mb_i);
            Yd[j] += one_minus_alpha * w_ji * delta;
            Rd[j] = apply_step(Yd[j]) - Zd[j];
            bool cross;
            if (tm == TH_ROWSUM) {
                cross = (Rd[j] > ub_d) || (Rd[j] < lb_d);
            } else {
                cross = std::fabs(Rd[j]) > base_thr * deg_1mb[(size_t)j];
            }
            if (!inq[(size_t)j] && cross) {
                q.push(j);
                inq[(size_t)j] = 1;
            }
        }
        work += (double)deg_i;
    }
    return work;
}

void NonlinearPPR::seed_active_set(double eps) {
    const ThresholdMode tm = threshold_mode;
    double base_thr = (1.0 - K * (1.0 - alpha)) * eps;
    #pragma omp parallel for schedule(static)
    for (int d = 0; d < F; ++d) {
        auto& q = active_queue[(size_t)d];
        auto& inq = in_queue[(size_t)d];
        const double* Rd = R[(size_t)d].data();
        double ub_d = (tm == TH_ROWSUM) ? alpha * rowsum_pos[(size_t)d] * eps : 0.0;
        double lb_d = (tm == TH_ROWSUM) ? alpha * rowsum_neg[(size_t)d] * eps : 0.0;
        for (int32_t i = 0; i < (int32_t)n; ++i) {
            bool cross;
            if (tm == TH_ROWSUM) {
                cross = (Rd[i] > ub_d) || (Rd[i] < lb_d);
            } else {
                cross = std::fabs(Rd[i]) > base_thr * deg_1mb[(size_t)i];
            }
            if (cross) {
                q.push(i);
                inq[(size_t)i] = 1;
            }
        }
    }
}

double NonlinearPPR::cleanup(double eps) {
    if (state_dirty) {
        seed_active_set(eps);
        state_dirty = false;
    }
    update_w.clear();
    for (int d = 0; d < F; ++d) {
        if (!active_queue[(size_t)d].empty()) update_w.push_back(d);
    }
    double total = 0.0;
    int W = (int)update_w.size();
    #pragma omp parallel for reduction(+ : total) schedule(dynamic)
    for (int k = 0; k < W; ++k) {
        total += cleanup_dim(update_w[k], eps);
    }
    return total;
}

void NonlinearPPR::update_edge_state_per_dim(int d, int32_t u, int32_t v,
                                             int sigma,
                                             int du_old, int dv_old,
                                             int du_new, int dv_new) {
    double* Zd = Z[(size_t)d].data();
    double* Yd = Y[(size_t)d].data();
    double* Rd = R[(size_t)d].data();
    double* Xd = X[(size_t)d].data();

    double du_1mb     = std::pow((double)du_old, 1.0 - beta);
    double dv_1mb     = std::pow((double)dv_old, 1.0 - beta);
    double du_b       = std::pow((double)du_old, beta);
    double dv_b       = std::pow((double)dv_old, beta);
    double du_new_1mb = std::pow((double)du_new, 1.0 - beta);
    double dv_new_1mb = std::pow((double)dv_new, 1.0 - beta);
    double du_new_b   = std::pow((double)du_new, beta);
    double dv_new_b   = std::pow((double)dv_new, beta);

    double x_u = Zd[u] / du_1mb;
    double x_v = Zd[v] / dv_1mb;
    double S_u = du_b / (1.0 - alpha) * (Yd[u] - alpha * Xd[u]);
    double S_v = dv_b / (1.0 - alpha) * (Yd[v] - alpha * Xd[v]);

    Zd[u] = (du_new_1mb / du_1mb) * Zd[u];
    Zd[v] = (dv_new_1mb / dv_1mb) * Zd[v];
    Yd[u] = alpha * Xd[u] + (1.0 - alpha) * (S_u + sigma * x_v) / du_new_b;
    Yd[v] = alpha * Xd[v] + (1.0 - alpha) * (S_v + sigma * x_u) / dv_new_b;
    Rd[u] = apply_step(Yd[u]) - Zd[u];
    Rd[v] = apply_step(Yd[v]) - Zd[v];
}

double NonlinearPPR::snapshot_operation(const int32_t* events, int64_t M,
                                        double eps) {
    const ThresholdMode tm = threshold_mode;
    double base_thr = (1.0 - K * (1.0 - alpha)) * eps;
    double r = 0;

    for (int64_t i = 0; i < M; ++i) {
        int32_t u = events[3 * i];
        int32_t v = events[3 * i + 1];
        int sigma = (int)events[3 * i + 2];

        int du_old = g.deg[(size_t)u];
        int dv_old = g.deg[(size_t)v];

        bool changed = false;
        if (sigma > 0) changed = g.insertEdge(u, v);
        else           changed = g.deleteEdge(u, v);
        if (!changed) continue;

        int du_new = g.deg[(size_t)u];
        int dv_new = g.deg[(size_t)v];

        refresh_degree_cache(u);
        if (u != v) refresh_degree_cache(v);

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int d = 0; d < F; ++d) {
            update_edge_state_per_dim(d, u, v, sigma,
                                      du_old, dv_old, du_new, dv_new);

            // R[u], R[v] just changed; enqueue if they cross threshold.
            auto& q = active_queue[(size_t)d];
            auto& inq = in_queue[(size_t)d];
            const auto& Rd = R[(size_t)d];
            double ub_d = 0.0, lb_d = 0.0;
            if (tm == TH_ROWSUM) {
                ub_d = alpha * rowsum_pos[(size_t)d] * eps;
                lb_d = alpha * rowsum_neg[(size_t)d] * eps;
            }
            auto cross = [&](int32_t j) -> bool {
                if (tm == TH_ROWSUM)
                    return Rd[(size_t)j] > ub_d || Rd[(size_t)j] < lb_d;
                return std::fabs(Rd[(size_t)j]) >
                       base_thr * deg_1mb[(size_t)j];
            };
            if (!inq[(size_t)u] && cross(u)) {
                q.push(u);
                inq[(size_t)u] = 1;
            }
            if (u != v) {
                if (!inq[(size_t)v] && cross(v)) {
                    q.push(v);
                    inq[(size_t)v] = 1;
                }
            }
        }
        r = cleanup(eps);
    }
    return r;
}

double NonlinearPPR::snapshot_operation_batched(const int32_t* events,
                                                int64_t M, double eps) {
    const ThresholdMode tm = threshold_mode;
    double base_thr = (1.0 - K * (1.0 - alpha)) * eps;

    struct EdgeChange { int32_t v; int8_t sigma; };

    // 1. One pass over events: record each affected node's pre-snapshot
    //    degree, then apply the graph edit, then record the (v, sigma)
    //    change on both endpoints' change lists.
    std::vector<int32_t> affected;
    affected.reserve((size_t)(2 * M));
    std::vector<int32_t> old_deg;            // parallel to `affected`
    old_deg.reserve((size_t)(2 * M));
    std::vector<std::vector<EdgeChange>> changes;
    changes.reserve((size_t)(2 * M));

    auto touch = [&](int32_t u) -> int32_t {
        int32_t idx = affected_idx_buf[(size_t)u];
        if (idx < 0) {
            idx = (int32_t)affected.size();
            affected_idx_buf[(size_t)u] = idx;
            affected.push_back(u);
            old_deg.push_back(g.deg[(size_t)u]);
            changes.emplace_back();
        }
        return idx;
    };

    for (int64_t i = 0; i < M; ++i) {
        int32_t u = events[3 * i];
        int32_t v = events[3 * i + 1];
        int sigma = (int)events[3 * i + 2];

        int32_t iu = touch(u);
        int32_t iv = (u == v) ? iu : touch(v);

        bool changed = (sigma > 0) ? g.insertEdge(u, v)
                                   : g.deleteEdge(u, v);
        if (!changed) continue;

        changes[(size_t)iu].push_back({v, (int8_t)sigma});
        if (u != v) changes[(size_t)iv].push_back({u, (int8_t)sigma});
    }

    size_t A = affected.size();

    // 2. Precompute old/new d^(1-beta), d^beta for each affected node, and
    //    refresh the per-node degree-power cache.
    std::vector<double> old_1mb(A), old_b(A), new_1mb(A), new_b(A);
    for (size_t k = 0; k < A; ++k) {
        int32_t u = affected[k];
        double dold = (double)old_deg[k];
        old_1mb[k] = std::pow(dold, 1.0 - beta);
        old_b[k]   = std::pow(dold, beta);
        refresh_degree_cache(u);
        new_1mb[k] = deg_1mb[(size_t)u];
        new_b[k]   = deg_b[(size_t)u];
    }

    // 3. One OMP region over dims. For each dim, snapshot x = Z/d_old^(1-beta)
    //    across all affected nodes, then do the per-node Z/Y/R update using
    //    the cached x's (so reads and writes don't race within the dim).
    #pragma omp parallel for schedule(static)
    for (int d = 0; d < F; ++d) {
        double* Zd = Z[(size_t)d].data();
        double* Yd = Y[(size_t)d].data();
        double* Rd = R[(size_t)d].data();
        const double* Xd = X[(size_t)d].data();
        auto& q = active_queue[(size_t)d];
        auto& inq = in_queue[(size_t)d];

        std::vector<double> x_cache(A);
        for (size_t k = 0; k < A; ++k) {
            x_cache[k] = Zd[(size_t)affected[k]] / old_1mb[k];
        }

        for (size_t k = 0; k < A; ++k) {
            int32_t u = affected[k];
            double du_old_1mb = old_1mb[k];
            double du_old_b   = old_b[k];
            double du_new_1mb = new_1mb[k];
            double du_new_b   = new_b[k];

            double S_u = du_old_b *
                         (Yd[(size_t)u] - alpha * Xd[(size_t)u]) /
                         (1.0 - alpha);
            double delta_S = 0.0;
            const auto& ch = changes[k];
            for (size_t e = 0; e < ch.size(); ++e) {
                int32_t vn = ch[e].v;
                int32_t vi = affected_idx_buf[(size_t)vn];
                double x_v;
                if (vi >= 0) {
                    x_v = x_cache[(size_t)vi];
                } else {
                    // Non-affected v: its degree (and d^(1-beta)) is
                    // unchanged across this snapshot, so deg_1mb[v] is
                    // the old value we want.
                    x_v = Zd[(size_t)vn] / deg_1mb[(size_t)vn];
                }
                delta_S += (double)ch[e].sigma * x_v;
            }

            Zd[(size_t)u] = (du_new_1mb / du_old_1mb) * Zd[(size_t)u];
            Yd[(size_t)u] = alpha * Xd[(size_t)u] +
                            (1.0 - alpha) * (S_u + delta_S) / du_new_b;
            Rd[(size_t)u] = apply_step(Yd[(size_t)u]) - Zd[(size_t)u];

            bool cross;
            if (tm == TH_ROWSUM) {
                double ub_d = alpha * rowsum_pos[(size_t)d] * eps;
                double lb_d = alpha * rowsum_neg[(size_t)d] * eps;
                cross = Rd[(size_t)u] > ub_d || Rd[(size_t)u] < lb_d;
            } else {
                cross = std::fabs(Rd[(size_t)u]) > base_thr * du_new_1mb;
            }
            if (!inq[(size_t)u] && cross) {
                q.push(u);
                inq[(size_t)u] = 1;
            }
        }
    }

    // 4. Clear affected_idx_buf entries we touched.
    for (size_t k = 0; k < A; ++k) {
        affected_idx_buf[(size_t)affected[k]] = -1;
    }

    return cleanup(eps);
}

void NonlinearPPR::apply_edge_events(const int32_t* events, int64_t M) {
    for (int64_t i = 0; i < M; ++i) {
        int32_t u = events[3 * i];
        int32_t v = events[3 * i + 1];
        int sigma = (int)events[3 * i + 2];
        bool changed;
        if (sigma > 0) changed = g.insertEdge(u, v);
        else           changed = g.deleteEdge(u, v);
        if (!changed) continue;
        refresh_degree_cache(u);
        if (u != v) refresh_degree_cache(v);
    }
}

void NonlinearPPR::get_z(double* out) const {
    for (int d = 0; d < F; ++d) {
        std::memcpy(out + (size_t)d * (size_t)n, Z[(size_t)d].data(),
                    sizeof(double) * (size_t)n);
    }
}
void NonlinearPPR::get_y(double* out) const {
    for (int d = 0; d < F; ++d) {
        std::memcpy(out + (size_t)d * (size_t)n, Y[(size_t)d].data(),
                    sizeof(double) * (size_t)n);
    }
}
void NonlinearPPR::get_r(double* out) const {
    for (int d = 0; d < F; ++d) {
        std::memcpy(out + (size_t)d * (size_t)n, R[(size_t)d].data(),
                    sizeof(double) * (size_t)n);
    }
}

std::vector<int32_t> NonlinearPPR::degrees() const { return g.deg; }

double NonlinearPPR::total_residual_l1() const {
    double s = 0.0;
    for (int d = 0; d < F; ++d) {
        const auto& Rd = R[(size_t)d];
        for (size_t i = 0; i < (size_t)n; ++i) s += std::fabs(Rd[i]);
    }
    return s;
}

} // namespace nlppr
