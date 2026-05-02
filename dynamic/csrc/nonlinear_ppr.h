#pragma once
#include "graph.h"
#include <queue>
#include <vector>
#include <cstdint>

namespace nlppr {

enum StepFn {
    SF_IDENTITY  = 0,
    SF_SIGMOID   = 1,   // K = 1/4
    SF_TANH      = 2,   // K = 1
    SF_SOFTPLUS  = 3,   // K = 1
    SF_RELU      = 4,   // K = 1
    SF_CLAMP_SYM = 5,   // clamp(x, -p, p); K = 1
    SF_STANH     = 6,   // tanh(p*x)/p; K = 1
    SF_LEAKY05   = 7,   // leaky_relu with slope 0.5 on negatives; K = 1
    SF_SHTANH    = 8,   // tanh(x - p); K = 1
    SF_HTANH     = 9    // hard tanh: clamp(x, -p, p), p = width; K = 1
};

enum ThresholdMode {
    // (1 - K(1-alpha)) * eps * d(i)^(1-beta); depends on both node degree and
    // step-fn Lipschitz constant. This is what the Algorithm-1 convergence
    // analysis uses.
    TH_DEGREE = 0,
    // Fixed per-dimension band.
    //   R[d][j] in (rowsum_neg[d] * eps, rowsum_pos[d] * eps)
    // independent of j's degree. rowsum_{pos,neg}[d] are sums of positive
    // and negative entries of X[d,:] respectively. Much looser in practice
    // for StandardScaler'd features.
    TH_ROWSUM = 1
};

class NonlinearPPR {
public:
    Graph g;
    int64_t n;
    int F;
    double alpha;
    double beta;
    double K;
    StepFn step_fn;
    double step_param;
    ThresholdMode threshold_mode;

    // Per-dim positive-/negative-entry sums of X. Used by TH_ROWSUM; populated
    // by set_features. Size F. Not used by TH_DEGREE.
    std::vector<double> rowsum_pos;
    std::vector<double> rowsum_neg;  // stored as negative numbers

    // Per-dim storage: outer index is feature dim (size F), inner vector is
    // size n. Each dimension's buffers are
    // independent and cheap to pass to a per-dim worker.
    std::vector<std::vector<double>> X;  // raw features s
    std::vector<std::vector<double>> Z;  // propagated z
    std::vector<std::vector<double>> Y;  // y = alpha*s + (1-alpha)*W*z
    std::vector<std::vector<double>> R;  // r = U(y) - z

    // Per-node degree-power caches; updated only when a node's degree changes.
    std::vector<double> deg_1mb;  // d(i)^(1-beta)
    std::vector<double> deg_b;    // d(i)^(beta)

    // Persistent per-dim active set so cleanup avoids an O(n) scan per call.
    std::vector<std::queue<int32_t>> active_queue;  // size F, FIFO per dim
    std::vector<std::vector<uint8_t>> in_queue;      // size F, each size n
    std::vector<int> update_w;  // dims with non-empty queue at cleanup entry
    bool state_dirty;  // true => next cleanup must do a full O(n) seed scan

    // Scratch buffer reused by snapshot_operation_batched: node id -> index
    // into this call's `affected` vector, or -1 if not affected. Kept across
    // calls (reset only at touched entries) to avoid an O(n) alloc each call.
    std::vector<int32_t> affected_idx_buf;

    NonlinearPPR(int64_t n, int F, double alpha, double beta, double K,
                 int step_fn_code, double step_param,
                 int threshold_mode_code = 0);

    void set_features(const double* X_in);  // [F, N]

    // Insert edges into graph (on top of the permanent self-loops already
    // installed by the ctor), initialize (z=0, y=alpha*s, r=U(y)), then
    // cleanup. Returns total weighted push work.
    double initial_operation(const int32_t* edges, int64_t E, double eps);

    // Apply edge events (insertion/deletion) one at a time using the
    // Algorithm-1 UpdateEdge, then cleanup. events[i] = (u, v, sigma) with
    // sigma = +1 for insert, -1 for delete.
    double snapshot_operation(const int32_t* events, int64_t M, double eps);

    // Same effect as snapshot_operation but batched:
    // first flush every edge edit into the graph, collect the set of
    // affected nodes, then do a single pass per affected node per dim
    // (one OMP region over dims instead of one per edge, pows hoisted
    // outside the dim loop).
    double snapshot_operation_batched(const int32_t* events, int64_t M,
                                      double eps);

    // Mutate only the graph, without touching (z, y, r). Used by baselines
    // that want the graph to advance without any dynamic maintenance.
    void apply_edge_events(const int32_t* events, int64_t M);

    // Reset (z=0, y=alpha*s, r=U(y)) on the current graph.
    void reset_state();

    // Run Cleanup on (current graph, current state). Returns total weighted
    // push work (sum of degrees touched).
    double cleanup(double eps);

    // Copy state out to dense [F, N] buffers.
    void get_z(double* out) const;
    void get_y(double* out) const;
    void get_r(double* out) const;

    std::vector<int32_t> degrees() const;
    double total_residual_l1() const;

    // Exposed for tests / debugging.
    double apply_step(double x) const;
    double threshold_i(int32_t i, double eps) const;

private:
    void update_edge_state_per_dim(int d, int32_t u, int32_t v, int sigma,
                                   int du_old, int dv_old,
                                   int du_new, int dv_new);
    double cleanup_dim(int d, double eps);
    void refresh_degree_cache(int32_t i);
    void seed_active_set(double eps);
};

} // namespace nlppr
