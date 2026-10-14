#include "ns3/core-module.h"
#include "ns3/opengym-module.h"

#ifdef WITH_ZMQ
#  include <zmq.hpp>
#  define ZMQ_AVAILABLE 1
#else
#  define ZMQ_AVAILABLE 0
#endif

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("GnbScheduler");

static const uint32_t N_CELLS = 3;
static const uint32_t N_SL    = 4;

// Lagrange multiplier settings
static const double LAGRANGE_INIT        = 0.10;
static const double LAGRANGE_MAX         = 3.00;
static const double LAGRANGE_LR_LEND_NC  = 0.80;
static const double LAGRANGE_LR_LEND_HO  = 0.50;
static const double LAGRANGE_LR_BORR_HO  = 0.60;

// Target thresholds for Lagrange constraints
static const double LEND_NC_TARGET  = 0.07;
static const double LEND_HO_TARGET  = 0.06;
static const double BORR_HO_TARGET  = 0.06;

// Capacity and slice settings
static const double CAPS [N_SL] = {0.23, 0.38, 0.13, 0.26};
static const double SMINS[N_SL] = {0.01, 0.01, 0.01, 0.01};
static const double SLICE_ALPHA[N_SL]       = {0.50, 1.00, 0.0, 0.0};
static const bool   IS_PRIORITY_SLICE[N_SL] = {true, true, false, false};
static const double ALPHA_SUM          = 1.5;
static const double BORROW_FRAC_TELEOP = 1.0 / 1.5;
static const double BORROW_FRAC_AUTO   = 0.5 / 1.5;
static const double UTIL_TARGET = 0.99;
static const double P_H = 1.0;
static const double P_N = 0.60;

static const double PRIO_SAT_DENOM = ALPHA_SUM * (P_H + P_N);

static const double HO_BLOCK_THRESH = 0.06;
static const double NC_BLOCK_THRESH = 0.03;
static const double HO_HARD_CAP     = 0.10;
static const double NC_HARD_CAP     = 0.10;

static const double LP_SAFETY_FRAC_DONATE = 0.40;
static const double DONATION_BOOST_MAX    = 1.4;
static const double NC_BUDGET_BASE = 0.78;

static const double ALLOC_MAX[N_SL] = {0.50, 0.65, 0.20, 0.45};
static const double ALLOC_MIN[N_SL] = {0.05, 0.10, 0.08, 0.09};

static std::string g_tickDir     = "bridge/tick_reports";
static uint32_t    g_simTime     = 200;
static uint32_t    g_gymPort     = 5555;
static uint32_t    g_curTick     = 0;
static uint32_t    g_curEp       = 0;
static bool        g_zmqPull     = false;
static uint32_t    g_arrivalPort = 5556;
static std::string g_bridgeHost  = "localhost";
static double      g_lambdaArrival = 45.0;

#if ZMQ_AVAILABLE
static zmq::context_t *g_zmqCtx   = nullptr;
static zmq::socket_t  *g_pullSock = nullptr;
static zmq::socket_t  *g_ackPush  = nullptr;
#endif

struct Flow {
    double bw_abs;
    int    slice_idx;
    bool   is_ho;
    int    remaining;
    int    acc_tick;
};

struct Cell {
    double load[N_SL]  = {0.92, 0.95, 0.92, 0.90};
    double alloc[N_SL] = {0.25, 0.40, 0.10, 0.25};
    double used[N_SL]  = {};
    double sinr_db     = 20.0;   // fixed, no longer updated from neighbours

    int ho_arr = 0, nc_arr = 0;
    int ho_acc = 0, ho_rej = 0, nc_acc = 0, nc_rej = 0;
    int ho_acc_ps[N_SL] = {}, ho_rej_ps[N_SL] = {};
    int nc_acc_ps[N_SL] = {}, nc_rej_ps[N_SL] = {};
    int sl_ho_acc[N_SL] = {}, sl_ho_rej[N_SL] = {};
    int sl_nc_acc[N_SL] = {}, sl_nc_rej[N_SL] = {};

    int sl_ho_acc_ep[N_SL] = {};
    int sl_ho_rej_ep[N_SL] = {};
    int sl_nc_acc_ep[N_SL] = {};
    int sl_nc_rej_ep[N_SL] = {};

    int ho_acc_ep = 0, ho_rej_ep = 0;
    int nc_acc_ep = 0, nc_rej_ep = 0;
    int ho_rej_lender_ep = 0, nc_rej_lender_ep = 0;

    double util_served_sum = 0.0;
    int    util_ticks      = 0;

    bool   overloaded  = false;
    bool   is_lender   = false;
    bool   is_borrower = false;

    bool   is_lender_ep   = false;
    bool   is_borrower_ep = false;

    bool   is_lender_prev   = false;
    bool   is_borrower_prev = false;

    double donation_fracs[N_SL]   = {};
    bool   slice_overloaded[N_SL] = {};

    double nc_budget_frac = NC_BUDGET_BASE;
    double ho_blk_ema = 0.0;
    double nc_blk_ema = 0.0;
    static constexpr double EMA_ALPHA = 0.25;

    // Lagrange state
    double lagrange_lend_nc = LAGRANGE_INIT;
    double lagrange_lend_ho = LAGRANGE_INIT;
    double lagrange_borr_ho = LAGRANGE_INIT;

    double lend_nc_blk_ema = 0.0;
    double lend_ho_blk_ema = 0.0;
    double borr_ho_blk_ema = 0.0;

    std::vector<Flow> active_flows;

    void RecomputeUsed() {
        for (uint32_t s = 0; s < N_SL; s++) used[s] = 0.0;
        for (auto &f : active_flows)
            used[f.slice_idx] += f.bw_abs;
    }

    void AgeFlows() {
        std::vector<Flow> alive;
        alive.reserve(active_flows.size());
        for (auto &f : active_flows) {
            f.remaining--;
            if (f.remaining > 0) alive.push_back(f);
        }
        active_flows = std::move(alive);
    }

    bool Admit(double bw_abs, int slice_idx, bool is_ho, int remaining, int tick) {
        RecomputeUsed();
        double spare = alloc[slice_idx] - used[slice_idx];
        if (spare >= bw_abs - 1e-8) {
            Flow f;
            f.bw_abs    = bw_abs;
            f.slice_idx = slice_idx;
            f.is_ho     = is_ho;
            f.remaining = remaining;
            f.acc_tick  = tick;
            active_flows.push_back(f);
            used[slice_idx] += bw_abs;
            return true;
        }
        return false;
    }

    bool AdmitHOWithFallback(double bw_abs, int slice_idx, int remaining, int tick) {
        if (Admit(bw_abs, slice_idx, true, remaining, tick)) return true;
        int fb = (slice_idx == 1) ? 0 : 1;
        double spare = alloc[fb] - used[fb];
        if (spare >= bw_abs - 1e-8) {
            Flow f;
            f.bw_abs    = bw_abs;
            f.slice_idx = fb;
            f.is_ho     = true;
            f.remaining = remaining;
            f.acc_tick  = tick;
            active_flows.push_back(f);
            used[fb] += bw_abs;
            return true;
        }
        return false;
    }

    bool AdmitHOForceful(double bw_abs, int slice_idx, int remaining, int tick) {
        if (AdmitHOWithFallback(bw_abs, slice_idx, remaining, tick)) return true;

        static const int PREEMPT_SLICES[N_SL] = {2, 3, 0, 1};
        for (int ps : PREEMPT_SLICES) {
            for (int attempt = 0; attempt < 3; attempt++) {
                int evict_idx = -1;
                double smallest_bw = 1e9;
                for (size_t i = 0; i < active_flows.size(); i++) {
                    if (!active_flows[i].is_ho &&
                        active_flows[i].slice_idx == ps &&
                        active_flows[i].bw_abs < smallest_bw) {
                        smallest_bw = active_flows[i].bw_abs;
                        evict_idx = (int)i;
                    }
                }
                if (evict_idx < 0) break;
                active_flows.erase(active_flows.begin() + evict_idx);
                RecomputeUsed();
                if (AdmitHOWithFallback(bw_abs, slice_idx, remaining, tick))
                    return true;
            }
        }
        return false;
    }

    bool AdmitNC(double bw_abs, int slice_idx, int remaining, int tick,
                 double ho_reserve = 0.0) {
        RecomputeUsed();
        double effective_cap = alloc[slice_idx] - ho_reserve;
        double spare = effective_cap - used[slice_idx];
        if (spare >= bw_abs - 1e-8) {
            Flow f;
            f.bw_abs    = bw_abs;
            f.slice_idx = slice_idx;
            f.is_ho     = false;
            f.remaining = remaining;
            f.acc_tick  = tick;
            active_flows.push_back(f);
            used[slice_idx] += bw_abs;
            return true;
        }
        return false;
    }
};

static Cell g_cells[N_CELLS];

static std::string ReadFile(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}
static double JsonDouble(const std::string &js, const std::string &key, double def = 0.0) {
    auto p = js.find("\"" + key + "\"");
    if (p == std::string::npos) return def;
    auto c = js.find(':', p);
    if (c == std::string::npos) return def;
    try { return std::stod(js.substr(c + 1)); }
    catch (...) { return def; }
}
static int JsonInt(const std::string &js, const std::string &key, int def = 0) {
    auto p = js.find("\"" + key + "\"");
    if (p == std::string::npos) return def;
    auto c = js.find(':', p);
    if (c == std::string::npos) return def;
    try { return std::stoi(js.substr(c + 1)); }
    catch (...) { return def; }
}
static std::string CellBlock(const std::string &js, int cell_id) {
    std::string key = "\"" + std::to_string(cell_id) + "\"";
    auto p = js.find(key);
    if (p == std::string::npos) return "";
    auto lb = js.find('{', p);
    if (lb == std::string::npos) return "";
    int depth = 0; size_t i = lb;
    for (; i < js.size(); i++) {
        if      (js[i] == '{') depth++;
        else if (js[i] == '}') { if (--depth == 0) break; }
    }
    return js.substr(lb, i - lb + 1);
}
struct Arrival {
    double bw_abs;
    bool   is_ho;
    int    slice_idx;
    int    remaining;
};
static std::vector<Arrival> ParseArrivals(const std::string &block) {
    std::vector<Arrival> out;
    auto p = block.find("\"arrivals\"");
    if (p == std::string::npos) return out;
    auto lb = block.find('[', p);
    auto rb = block.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return out;
    std::string arr = block.substr(lb + 1, rb - lb - 1);
    size_t pos = 0;
    while ((pos = arr.find('{', pos)) != std::string::npos) {
        auto end = arr.find('}', pos);
        if (end == std::string::npos) break;
        std::string obj = arr.substr(pos, end - pos + 1);
        pos = end + 1;
        Arrival a;
        a.bw_abs    = JsonDouble(obj, "bw", 0.006);
        a.slice_idx = std::max(0, std::min((int)N_SL - 1,
                               (int)JsonDouble(obj, "slice_idx", 3)));
        a.remaining = std::max(5, JsonInt(obj, "remaining", 5));
        auto ih = obj.find("\"is_ho\"");
        a.is_ho = (ih != std::string::npos &&
                   (obj.find("true", ih) != std::string::npos ||
                    obj.find("1",    ih) != std::string::npos));
        out.push_back(a);
    }
    return out;
}
static double ParseCellLoad(const std::string &block, int slice_idx) {
    auto p = block.find("\"cell_load\"");
    if (p != std::string::npos) {
        auto lb = block.find('[', p);
        auto rb = block.find(']', lb);
        if (lb != std::string::npos && rb != std::string::npos) {
            std::string a = block.substr(lb + 1, rb - lb - 1);
            std::istringstream ss(a); std::string tok; int idx = 0;
            while (std::getline(ss, tok, ',')) {
                if (idx == slice_idx)
                    try { return std::min(std::stod(tok), 1.0); }
                    catch (...) { break; }
                idx++;
            }
        }
    }
    return 0.92;
}

static void ApplyTickJson(const std::string &js) {
    auto cp = js.find("\"cells\"");
    if (cp == std::string::npos) return;
    std::string cells_sec = js.substr(cp);

    for (uint32_t c = 0; c < N_CELLS; c++) {
        Cell &cs = g_cells[c];

        cs.ho_arr = cs.nc_arr = cs.ho_acc = cs.ho_rej =
        cs.nc_acc = cs.nc_rej = 0;

        cs.overloaded = false;

        for (uint32_t s = 0; s < N_SL; s++) {
            cs.ho_acc_ps[s] = cs.ho_rej_ps[s] = 0;
            cs.nc_acc_ps[s] = cs.nc_rej_ps[s] = 0;
            cs.sl_ho_acc[s] = cs.sl_ho_rej[s] = 0;
            cs.sl_nc_acc[s] = cs.sl_nc_rej[s] = 0;
            cs.slice_overloaded[s] = false;
        }

        if (g_curTick == 0) {
            for (uint32_t s = 0; s < N_SL; s++)
                cs.alloc[s] = CAPS[s];
        }

        cs.AgeFlows();
        cs.RecomputeUsed();

        for (uint32_t s = 0; s < N_SL; s++) {
            if (cs.alloc[s] < cs.used[s])
                cs.alloc[s] = cs.used[s];
        }

        std::string blk = CellBlock(cells_sec, (int)c);
        if (blk.empty()) continue;

        for (uint32_t s = 0; s < N_SL; s++)
            cs.load[s] = ParseCellLoad(blk, (int)s);

        auto arrivals = ParseArrivals(blk);

        double ho_reservation[N_SL];
        for (uint32_t s = 0; s < N_SL; s++) {
            if (IS_PRIORITY_SLICE[s]) {
                double blk_ema = cs.ho_blk_ema;
                if      (blk_ema > 0.08) ho_reservation[s] = cs.alloc[s] * 0.03;
                else if (blk_ema > 0.03) ho_reservation[s] = cs.alloc[s] * 0.015;
                else                     ho_reservation[s] = cs.alloc[s] * 0.005;
            } else {
                ho_reservation[s] = 0.0;
            }
        }

        if (cs.is_borrower) {
            double extra_reserve = 0.0;
            if (cs.ho_blk_ema > 0.06)      extra_reserve = 0.06;
            else if (cs.ho_blk_ema > 0.03) extra_reserve = 0.03;
            for (uint32_t s = 0; s < N_SL; s++)
                ho_reservation[s] += extra_reserve;
        }

        // PASS 1: HO flows
        for (auto &a : arrivals) {
            if (!a.is_ho) continue;
            cs.ho_arr++;
            bool ok = cs.AdmitHOForceful(
                a.bw_abs, a.slice_idx, a.remaining, (int)g_curTick);
            if (ok) {
                cs.ho_acc++;
                cs.ho_acc_ps[a.slice_idx]++;
                cs.sl_ho_acc[a.slice_idx]++;
                cs.ho_acc_ep++;
                cs.sl_ho_acc_ep[a.slice_idx]++;
            } else {
                cs.ho_rej++;
                cs.ho_rej_ps[a.slice_idx]++;
                cs.sl_ho_rej[a.slice_idx]++;
                cs.ho_rej_ep++;
                cs.sl_ho_rej_ep[a.slice_idx]++;
                if (cs.is_lender && !IS_PRIORITY_SLICE[a.slice_idx])
                    cs.ho_rej_lender_ep++;
            }
        }

        {
            double ho_tot = (double)(cs.ho_acc + cs.ho_rej);
            double ho_blk = (ho_tot > 0) ? (double)cs.ho_rej / ho_tot : 0.0;
            if (ho_blk <= HO_BLOCK_THRESH) {
                cs.nc_budget_frac = NC_BUDGET_BASE;
            } else if (ho_blk >= 0.20) {
                cs.nc_budget_frac = 0.40;
            } else {
                double t = (ho_blk - HO_BLOCK_THRESH) / (0.20 - HO_BLOCK_THRESH);
                cs.nc_budget_frac = NC_BUDGET_BASE - t * (NC_BUDGET_BASE - 0.40);
            }
        }

        // PASS 2: NC priority slices
        cs.RecomputeUsed();
        static const int PRIORITY_ORDER[2] = {1, 0};
        for (int pi = 0; pi < 2; pi++) {
            int target_sl = PRIORITY_ORDER[pi];
            for (auto &a : arrivals) {
                if (a.is_ho) continue;
                if (a.slice_idx != target_sl) continue;
                cs.nc_arr++;
                bool ok = cs.AdmitNC(a.bw_abs, a.slice_idx, a.remaining,
                                     (int)g_curTick, ho_reservation[a.slice_idx]);
                if (ok) {
                    cs.nc_acc++;
                    cs.nc_acc_ps[a.slice_idx]++;
                    cs.sl_nc_acc[a.slice_idx]++;
                    cs.nc_acc_ep++;
                    cs.sl_nc_acc_ep[a.slice_idx]++;
                } else {
                    cs.nc_rej++;
                    cs.nc_rej_ps[a.slice_idx]++;
                    cs.sl_nc_rej[a.slice_idx]++;
                    cs.nc_rej_ep++;
                    cs.sl_nc_rej_ep[a.slice_idx]++;
                    cs.slice_overloaded[a.slice_idx] = true;
                }
            }
        }

        // PASS 2b: cross-fallback priority with HO/NC gap guard
        {
            cs.RecomputeUsed();
            static const int NC_FALLBACK[2] = {1, 0};
            for (size_t ai = 0; ai < arrivals.size(); ai++) {
                const Arrival &a = arrivals[ai];
                if (a.is_ho)                              continue;
                if (!IS_PRIORITY_SLICE[a.slice_idx])      continue;
                if (!cs.slice_overloaded[a.slice_idx])    continue;
                if (cs.nc_rej_ps[a.slice_idx] <= 0)       continue;

                double ho_tot_fb = (double)(cs.ho_acc + cs.ho_rej);
                double nc_tot_fb = (double)(cs.nc_acc + cs.nc_rej);
                double ho_acc_fb = ho_tot_fb > 0 ? (double)cs.ho_acc / ho_tot_fb : 1.0;
                double nc_acc_fb = nc_tot_fb > 0 ? (double)cs.nc_acc / nc_tot_fb : 0.0;
                if (ho_acc_fb - nc_acc_fb < 0.04) continue;

                int dst = NC_FALLBACK[a.slice_idx];
                double spare = cs.alloc[dst] - cs.used[dst] - ho_reservation[dst];
                if (spare < a.bw_abs - 1e-8)              continue;
                Flow f;
                f.bw_abs    = a.bw_abs;
                f.slice_idx = dst;
                f.is_ho     = false;
                f.remaining = a.remaining;
                f.acc_tick  = (int)g_curTick;
                cs.active_flows.push_back(f);
                cs.used[dst] += a.bw_abs;
                cs.nc_rej             = std::max(0, cs.nc_rej - 1);
                cs.nc_rej_ep          = std::max(0, cs.nc_rej_ep - 1);
                cs.nc_rej_ps[a.slice_idx] = std::max(0, cs.nc_rej_ps[a.slice_idx] - 1);
                cs.nc_acc++;
                cs.nc_acc_ep++;
            }
        }

        // PASS 3: NC non-priority slices
        cs.RecomputeUsed();
        for (auto &a : arrivals) {
            if (a.is_ho) continue;
            if (IS_PRIORITY_SLICE[a.slice_idx]) continue;
            cs.nc_arr++;
            bool ok = cs.AdmitNC(a.bw_abs, a.slice_idx, a.remaining,
                                 (int)g_curTick, 0.0);
            if (ok) {
                cs.nc_acc++;
                cs.nc_acc_ps[a.slice_idx]++;
                cs.sl_nc_acc[a.slice_idx]++;
                cs.nc_acc_ep++;
                cs.sl_nc_acc_ep[a.slice_idx]++;
            } else {
                cs.nc_rej++;
                cs.nc_rej_ps[a.slice_idx]++;
                cs.sl_nc_rej[a.slice_idx]++;
                cs.nc_rej_ep++;
                cs.sl_nc_rej_ep[a.slice_idx]++;
                if (cs.is_lender && cs.donation_fracs[a.slice_idx] > 1e-8)
                    cs.nc_rej_lender_ep++;
                cs.slice_overloaded[a.slice_idx] = true;
            }
        }

        // PASS 3b: cross-fallback non-priority
        {
            cs.RecomputeUsed();
            static const int NP_FALLBACK[4] = {-1, -1, 3, 2};
            int fallback_budget[N_SL] = {};
            for (uint32_t s = 0; s < N_SL; s++)
                fallback_budget[s] = cs.nc_rej_ps[s];

            for (auto &a : arrivals) {
                if (a.is_ho)                           continue;
                if (IS_PRIORITY_SLICE[a.slice_idx])    continue;
                if (!cs.slice_overloaded[a.slice_idx]) continue;
                if (fallback_budget[a.slice_idx] <= 0) continue;
                int dst = NP_FALLBACK[a.slice_idx];
                if (dst < 0)                           continue;
                double spare = cs.alloc[dst] - cs.used[dst];
                if (spare < a.bw_abs - 1e-8)           continue;
                Flow f;
                f.bw_abs    = a.bw_abs;
                f.slice_idx = dst;
                f.is_ho     = false;
                f.remaining = a.remaining;
                f.acc_tick  = (int)g_curTick;
                cs.active_flows.push_back(f);
                cs.used[dst] += a.bw_abs;
                cs.nc_rej    = std::max(0, cs.nc_rej - 1);
                cs.nc_rej_ep = std::max(0, cs.nc_rej_ep - 1);
                cs.nc_rej_ps[a.slice_idx] = std::max(0, cs.nc_rej_ps[a.slice_idx] - 1);
                cs.nc_acc++;
                cs.nc_acc_ep++;
                fallback_budget[a.slice_idx]--;
            }
        }

        cs.RecomputeUsed();
        cs.overloaded = cs.slice_overloaded[0] || cs.slice_overloaded[1];

        // PASS 4: emergency fallback
        {
            cs.RecomputeUsed();
            int pass4_budget[N_SL] = {};
            for (uint32_t s = 0; s < N_SL; s++)
                pass4_budget[s] = cs.nc_rej_ps[s];

            for (auto &a : arrivals) {
                if (a.is_ho)                           continue;
                if (!IS_PRIORITY_SLICE[a.slice_idx])   continue;
                if (!cs.slice_overloaded[a.slice_idx]) continue;
                if (pass4_budget[a.slice_idx] <= 0)    continue;
                for (int dst : {2, 3}) {
                    double spare = cs.alloc[dst] - cs.used[dst];
                    if (spare < a.bw_abs - 1e-8)       continue;
                    Flow f;
                    f.bw_abs    = a.bw_abs;
                    f.slice_idx = dst;
                    f.is_ho     = false;
                    f.remaining = a.remaining;
                    f.acc_tick  = (int)g_curTick;
                    cs.active_flows.push_back(f);
                    cs.used[dst] += a.bw_abs;
                    cs.nc_rej    = std::max(0, cs.nc_rej - 1);
                    cs.nc_rej_ep = std::max(0, cs.nc_rej_ep - 1);
                    cs.nc_rej_ps[a.slice_idx] = std::max(0, cs.nc_rej_ps[a.slice_idx] - 1);
                    cs.nc_acc++;
                    cs.nc_acc_ep++;
                    pass4_budget[a.slice_idx]--;
                    break;
                }
            }
        }

        cs.is_borrower = cs.overloaded;
        if (cs.is_borrower) cs.is_borrower_ep = true;

        double total_used = 0.0;
        for (uint32_t s = 0; s < N_SL; s++) total_used += cs.used[s];
        total_used = std::min(total_used, 1.0);
        cs.util_served_sum += total_used;
        cs.util_ticks++;

        double ho_tot = (double)(cs.ho_acc + cs.ho_rej);
        double nc_tot = (double)(cs.nc_acc + cs.nc_rej);
        double ho_blk = ho_tot > 0 ? (double)cs.ho_rej / ho_tot : 0.0;
        double nc_blk = nc_tot > 0 ? (double)cs.nc_rej / nc_tot : 0.0;
        cs.ho_blk_ema = Cell::EMA_ALPHA * ho_blk + (1.0 - Cell::EMA_ALPHA) * cs.ho_blk_ema;
        cs.nc_blk_ema = Cell::EMA_ALPHA * nc_blk + (1.0 - Cell::EMA_ALPHA) * cs.nc_blk_ema;

        // Lagrange update
        double lend_nc_blk_tick = 0.0;
        double lend_ho_blk_tick = 0.0;
        double borr_ho_blk_tick = 0.0;

        if (cs.is_lender) {
            int nt_donor = 0, nr_donor = 0;
            for (uint32_t s = 0; s < N_SL; s++) {
                if (!IS_PRIORITY_SLICE[s]) {
                    nt_donor += cs.nc_acc_ps[s] + cs.nc_rej_ps[s];
                    nr_donor += cs.nc_rej_ps[s];
                }
            }
            lend_nc_blk_tick = (nt_donor > 0) ? (double)nr_donor / nt_donor : 0.0;
            double lho_tot   = (double)(cs.ho_acc + cs.ho_rej);
            lend_ho_blk_tick = (lho_tot > 0) ? (double)cs.ho_rej / lho_tot : 0.0;
        }
        if (cs.is_borrower) {
            double bho_tot   = (double)(cs.ho_acc + cs.ho_rej);
            borr_ho_blk_tick = (bho_tot > 0) ? (double)cs.ho_rej / bho_tot : 0.0;
        }

        cs.lend_nc_blk_ema = Cell::EMA_ALPHA * lend_nc_blk_tick
                           + (1.0 - Cell::EMA_ALPHA) * cs.lend_nc_blk_ema;
        cs.lend_ho_blk_ema = Cell::EMA_ALPHA * lend_ho_blk_tick
                           + (1.0 - Cell::EMA_ALPHA) * cs.lend_ho_blk_ema;
        cs.borr_ho_blk_ema = Cell::EMA_ALPHA * borr_ho_blk_tick
                           + (1.0 - Cell::EMA_ALPHA) * cs.borr_ho_blk_ema;

        cs.lagrange_lend_nc = std::max(0.0, std::min(LAGRANGE_MAX,
            cs.lagrange_lend_nc
            + LAGRANGE_LR_LEND_NC * (cs.lend_nc_blk_ema - LEND_NC_TARGET)));
        cs.lagrange_lend_ho = std::max(0.0, std::min(LAGRANGE_MAX,
            cs.lagrange_lend_ho
            + LAGRANGE_LR_LEND_HO * (cs.lend_ho_blk_ema - LEND_HO_TARGET)));
        cs.lagrange_borr_ho = std::max(0.0, std::min(LAGRANGE_MAX,
            cs.lagrange_borr_ho
            + LAGRANGE_LR_BORR_HO * (cs.borr_ho_blk_ema - BORR_HO_TARGET)));
    }
    // ── REMOVED: inter-cell SINR update using neighbour loads ──
    // Each cell's sinr_db stays at its fixed default (20.0).
    // nb_load_left / nb_load_right fields also removed.
}

static bool PullTickZmq() {
#if ZMQ_AVAILABLE
    if (!g_pullSock) return false;
    zmq::message_t msg;
    auto res = g_pullSock->recv(msg, zmq::recv_flags::none);
    if (!res) return false;
    std::string js(static_cast<char *>(msg.data()), msg.size());
    ApplyTickJson(js);
    if (g_ackPush) {
        zmq::message_t ack(1);
        static_cast<char *>(ack.data())[0] = 'K';
        try { g_ackPush->send(ack, zmq::send_flags::dontwait); }
        catch (...) {}
    }
    return true;
#else
    return false;
#endif
}
static void LoadTickFile(uint32_t /*ep*/, uint32_t tick) {
    std::ostringstream fname;
    fname << g_tickDir
          << "/ep" << std::setw(4) << std::setfill('0') << (g_curEp % 300)
          << "_tick" << std::setw(4) << std::setfill('0') << tick << ".json";
    std::string js = ReadFile(fname.str());
    if (js.empty()) {
        std::ostringstream fn2;
        fn2 << g_tickDir << "/tick_"
            << std::setw(4) << std::setfill('0') << tick << ".json";
        js = ReadFile(fn2.str());
    }
    if (!js.empty()) ApplyTickJson(js);
}
static void LoadTick(uint32_t ep, uint32_t tick) {
    if (g_zmqPull && ZMQ_AVAILABLE) {
        if (!PullTickZmq()) LoadTickFile(ep, tick);
    } else {
        LoadTickFile(ep, tick);
    }
}
static void ZmqInit() {
#if ZMQ_AVAILABLE
    g_zmqCtx   = new zmq::context_t(1);
    g_pullSock = new zmq::socket_t(*g_zmqCtx, zmq::socket_type::pull);
    g_pullSock->set(zmq::sockopt::rcvtimeo, 10000);
    std::string pullAddr = "tcp://" + g_bridgeHost + ":" +
                           std::to_string(g_arrivalPort);
    g_pullSock->connect(pullAddr);
    NS_LOG_UNCOND("ZMQ PULL -> " << pullAddr);
    g_ackPush = new zmq::socket_t(*g_zmqCtx, zmq::socket_type::push);
    g_ackPush->set(zmq::sockopt::sndtimeo, 200);
    std::string ackAddr = "tcp://" + g_bridgeHost + ":" +
                          std::to_string(g_arrivalPort + 1);
    g_ackPush->connect(ackAddr);
#else
    g_zmqPull = false;
#endif
}
static void ZmqTeardown() {
#if ZMQ_AVAILABLE
    if (g_pullSock) { g_pullSock->close(); delete g_pullSock; g_pullSock=nullptr; }
    if (g_ackPush)  { g_ackPush->close();  delete g_ackPush;  g_ackPush =nullptr; }
    if (g_zmqCtx)   { g_zmqCtx->close();   delete g_zmqCtx;   g_zmqCtx  =nullptr; }
#endif
}

static void PrewarmFlows() {
    static const double SYNTHETIC_BW   = 0.007;
    static const int    SYNTHETIC_LIFE = 400;
    static const double FILL_TARGET[N_SL] = {0.75, 0.80, 0.65, 0.55};

    for (uint32_t c = 0; c < N_CELLS; c++) {
        Cell &cs = g_cells[c];
        for (uint32_t s = 0; s < N_SL; s++)
            cs.alloc[s] = CAPS[s];
        cs.active_flows.clear();

        for (uint32_t s = 0; s < N_SL; s++) {
            double target_used = CAPS[s] * FILL_TARGET[s];
            double filled = 0.0;
            int safety = 0;
            while (filled + SYNTHETIC_BW <= target_used - 1e-6 && safety++ < 1000) {
                Flow f;
                f.bw_abs    = SYNTHETIC_BW;
                f.slice_idx = s;
                f.is_ho     = false;
                f.remaining = SYNTHETIC_LIFE;
                f.acc_tick  = -1;
                cs.active_flows.push_back(f);
                filled += SYNTHETIC_BW;
            }
        }
        cs.RecomputeUsed();
    }
}

/* ==================== OpenGym Environment ==================== */

static const uint32_t OBS_PER_CELL = 22;

class GnbEnv : public OpenGymEnv {
public:
    GnbEnv() {}
    static TypeId GetTypeId() {
        static TypeId tid = TypeId("GnbEnv")
            .SetParent<OpenGymEnv>().SetGroupName("OpenGym");
        return tid;
    }

    Ptr<OpenGymSpace> GetObservationSpace() override {
        std::vector<uint32_t> shape = {N_CELLS * OBS_PER_CELL};
        return CreateObject<OpenGymBoxSpace>(
            0.0f, 1.0f, shape, TypeNameGet<float>());
    }

    Ptr<OpenGymSpace> GetActionSpace() override {
        std::vector<uint32_t> shape = {N_CELLS * 4u};
        return CreateObject<OpenGymBoxSpace>(
            0.0f, 1.0f, shape, TypeNameGet<float>());
    }

    Ptr<OpenGymDataContainer> GetObservation() override {
        std::vector<uint32_t> shape = {N_CELLS * OBS_PER_CELL};
        auto box = CreateObject<OpenGymBoxContainer<float>>(shape);

        for (uint32_t c = 0; c < N_CELLS; c++) {
            const Cell &cs = g_cells[c];

            for (uint32_t s = 0; s < N_SL; s++)
                box->AddValue((float)std::min(cs.load[s], 1.0));        // 0-3

            for (uint32_t s = 0; s < N_SL; s++) {
                double un = cs.alloc[s] > 1e-8
                            ? std::min(cs.used[s] / cs.alloc[s], 1.0) : 0.0;
                box->AddValue((float)un);                                // 4-7
            }

            for (uint32_t s = 0; s < N_SL; s++)
                box->AddValue((float)cs.alloc[s]);                      // 8-11

            box->AddValue((float)std::min(cs.sinr_db / 25.0, 1.0));    // 12
            box->AddValue((float)std::min(cs.ho_arr / 8.0, 1.0));      // 13
            box->AddValue(cs.is_borrower ? 1.0f : 0.0f);               // 14
            box->AddValue((float)std::min(cs.ho_blk_ema, 1.0));        // 15
            box->AddValue((float)std::min(cs.nc_blk_ema, 1.0));        // 16

            for (uint32_t s = 0; s < 2; s++) {                         // 17-18
                int nt = cs.sl_nc_acc[s] + cs.sl_nc_rej[s];
                double nr = (nt > 0) ? (double)cs.sl_nc_rej[s] / nt : 0.0;
                box->AddValue((float)std::min(nr, 1.0));
            }

            // Lagrange observations                                    // 19-21
            box->AddValue((float)std::min(cs.lagrange_lend_nc / LAGRANGE_MAX, 1.0));
            box->AddValue((float)std::min(cs.lagrange_borr_ho / LAGRANGE_MAX, 1.0));
            box->AddValue((float)std::min(cs.lend_nc_blk_ema, 1.0));
        }
        return box;
    }

    float GetReward() override {
        double total_reward = 0.0;

        for (uint32_t c = 0; c < N_CELLS; c++) {
            const Cell &cs = g_cells[c];

            double ho_tot   = (double)(cs.ho_acc + cs.ho_rej);
            double nc_tot   = (double)(cs.nc_acc + cs.nc_rej);
            double ho_acc_r = (ho_tot > 0) ? (double)cs.ho_acc / ho_tot : 1.0;
            double nc_acc_r = (nc_tot > 0) ? (double)cs.nc_acc / nc_tot : 1.0;
            double ho_blk   = 1.0 - ho_acc_r;
            double nc_blk   = 1.0 - nc_acc_r;

            double total_used = 0.0;
            for (uint32_t s = 0; s < N_SL; s++) total_used += cs.used[s];
            double util = std::min(total_used, 1.0);

            // R_util
            double R_util = 0.0;
            if (util <= 0.80) {
                R_util = 4.0 * util;
            } else if (util <= 0.90) {
                R_util = 4.0 * 0.80 + 12.0 * (util - 0.80);
            } else if (util <= 0.95) {
                R_util = 4.0 * 0.80 + 12.0 * 0.10 + 40.0 * (util - 0.90);
            } else if (util <= 0.98) {
                R_util = 4.0 * 0.80 + 12.0 * 0.10 + 40.0 * 0.05
                       + 80.0 * (util - 0.95);
            } else {
                R_util = 4.0 * 0.80 + 12.0 * 0.10 + 40.0 * 0.05
                       + 80.0 * 0.03 + 30.0 * (util - 0.98);
            }

            // R_ho
            double R_ho = 0.0;
            if (ho_tot > 0) {
                if (ho_acc_r >= 0.96) {
                    R_ho = 10.0;
                } else if (ho_acc_r >= 0.94) {
                    R_ho = 6.0 + 4.0 * (ho_acc_r - 0.94) / 0.02;
                } else if (ho_acc_r >= 0.88) {
                    R_ho = 6.0 * (ho_acc_r - 0.88) / 0.06;
                } else {
                    R_ho = -40.0 * (0.88 - ho_acc_r);
                }
            }

            // R_nc
            double R_nc = 0.0;
            if (nc_tot > 0) {
                if (nc_acc_r >= 0.93) {
                    R_nc = 3.0 * nc_acc_r;
                } else if (nc_acc_r >= 0.80) {
                    R_nc = 1.5 * nc_acc_r;
                } else {
                    R_nc = 0.8 * nc_acc_r
                         - 5.0 * std::max(0.0, 0.80 - nc_acc_r);
                }
            }

            // R_inversion
            double R_inversion = 0.0;
            if (ho_tot > 0 && nc_tot > 0) {
                double inv = nc_acc_r - ho_acc_r;
                if (inv > 0.05) {
                    R_inversion = -80.0 * inv * inv
                                - 8.0 * (inv - 0.05);
                } else if (inv > 0.0) {
                    R_inversion = -20.0 * inv * inv;
                }
            }

            // R_blk
            double R_blk = 0.0;
            if (ho_blk > HO_BLOCK_THRESH) {
                double v = ho_blk - HO_BLOCK_THRESH;
                R_blk -= 60.0 * v * v;
                if (ho_blk > HO_HARD_CAP)
                    R_blk -= 20.0 * (ho_blk - HO_HARD_CAP);
            }
            if (nc_blk > NC_BLOCK_THRESH) {
                double v = nc_blk - NC_BLOCK_THRESH;
                R_blk -= 55.0 * v * v;
                if (nc_blk > NC_HARD_CAP)
                    R_blk -= 25.0 * (nc_blk - NC_HARD_CAP);
            }

            // R_lend_nc: penalise lender when its LP-slice NC blocking rises
            double R_lend_nc = 0.0;
            if (cs.is_lender) {
                int nt_donor = 0, nr_donor = 0;
                for (uint32_t s = 0; s < N_SL; s++) {
                    if (!IS_PRIORITY_SLICE[s]) {
                        nt_donor += cs.nc_acc_ps[s] + cs.nc_rej_ps[s];
                        nr_donor += cs.nc_rej_ps[s];
                    }
                }
                double lend_nc_blk_now = (nt_donor > 0)
                    ? (double)nr_donor / nt_donor : 0.0;

                R_lend_nc -= 8.0 * lend_nc_blk_now;
                double v_nc = std::max(0.0, lend_nc_blk_now - LEND_NC_TARGET);
                R_lend_nc -= 120.0 * v_nc * v_nc;
                if (lend_nc_blk_now > 0.15)
                    R_lend_nc -= 40.0 * (lend_nc_blk_now - 0.15);
                R_lend_nc -= cs.lagrange_lend_nc
                           * std::max(0.0, lend_nc_blk_now - LEND_NC_TARGET);
                if (lend_nc_blk_now <= LEND_NC_TARGET)
                    R_lend_nc += 2.0 * (LEND_NC_TARGET - lend_nc_blk_now);
            }

            // R_lend_ho: penalise lender when its HO blocking rises
            double R_lend_ho = 0.0;
            if (cs.is_lender) {
                double lend_ho_blk_now = (ho_tot > 0)
                    ? (double)cs.ho_rej / ho_tot : 0.0;
                double v_ho = std::max(0.0, lend_ho_blk_now - LEND_HO_TARGET);
                R_lend_ho -= 5.0 * lend_ho_blk_now;
                R_lend_ho -= 80.0 * v_ho * v_ho;
                R_lend_ho -= cs.lagrange_lend_ho
                           * std::max(0.0, lend_ho_blk_now - LEND_HO_TARGET);
                if (lend_ho_blk_now <= LEND_HO_TARGET)
                    R_lend_ho += 1.0 * (LEND_HO_TARGET - lend_ho_blk_now);
            }

            // R_priority
            double R_priority = 0.0;
            for (uint32_t s = 0; s < N_SL; s++) {
                if (!IS_PRIORITY_SLICE[s]) continue;
                int ht = cs.sl_ho_acc[s] + cs.sl_ho_rej[s];
                int nt = cs.sl_nc_acc[s] + cs.sl_nc_rej[s];
                double hr = (ht > 0) ? (double)cs.sl_ho_acc[s] / ht : 1.0;
                double nr = (nt > 0) ? (double)cs.sl_nc_acc[s] / nt : 1.0;
                R_priority += SLICE_ALPHA[s] * (P_H * hr + P_N * nr)
                            / (P_H + P_N);
            }

            // R_slice_util
            double R_slice_util = 0.0;
            for (uint32_t s = 0; s < N_SL; s++) {
                double slice_util = cs.alloc[s] > 1e-8
                    ? std::min(cs.used[s] / cs.alloc[s], 1.0) : 0.0;
                double w = IS_PRIORITY_SLICE[s] ? 2.5 : 0.8;
                R_slice_util += w * slice_util;
            }

            // R_starvation
            double R_starvation = 0.0;
            for (uint32_t s = 0; s < N_SL; s++) {
                if (IS_PRIORITY_SLICE[s]) continue;
                double min_safe = (s == 2) ? 0.08 : 0.09;
                if (cs.alloc[s] < min_safe) {
                    R_starvation -= 15.0 * (min_safe - cs.alloc[s]) / min_safe;
                }
            }

            double cr = (1.20 * R_util
                       + 0.80 * R_ho
                       + 0.70 * R_nc
                       + 0.30 * R_inversion
                       + 1.00 * R_blk
                       + 1.50 * R_lend_nc
                       + 1.00 * R_lend_ho
                       + 0.50 * R_priority
                       + 0.30 * R_slice_util
                       + 0.80 * R_starvation);

            cr = std::max(-15.0, std::min(15.0, cr));
            total_reward += cr;
        }

        return (float)(total_reward / N_CELLS);
    }

    bool GetGameOver() override {
        return g_curTick >= g_simTime;
    }

    std::string GetExtraInfo() override {
        std::ostringstream o;
        o << std::fixed << std::setprecision(4);
        o << "{";
        for (uint32_t c = 0; c < N_CELLS; c++) {
            const Cell &cs = g_cells[c];
            int ht = cs.ho_acc_ep + cs.ho_rej_ep;
            int nt = cs.nc_acc_ep + cs.nc_rej_ep;
            double util = cs.util_ticks > 0
                          ? cs.util_served_sum / cs.util_ticks
                          : 0.0;
            double ho_rate = ht > 0 ? (double)cs.ho_acc_ep / ht : 1.0;
            double nc_rate = nt > 0 ? (double)cs.nc_acc_ep / nt : 1.0;
            double ho_blk  = ht > 0 ? (double)cs.ho_rej_ep / ht : 0.0;
            double nc_blk  = nt > 0 ? (double)cs.nc_rej_ep / nt : 0.0;
            double ho_drop_borrower = cs.is_borrower_ep ? ho_blk : 0.0;
            double nc_blk_lender    = cs.is_lender_ep   ? nc_blk : 0.0;

            int nt_donor = 0;
            int nr_donor = 0;
            for (uint32_t s = 0; s < N_SL; s++) {
                if (!IS_PRIORITY_SLICE[s]) {
                    nt_donor += cs.sl_nc_acc_ep[s] + cs.sl_nc_rej_ep[s];
                    nr_donor += cs.sl_nc_rej_ep[s];
                }
            }
            double lender_nc_blk = (cs.is_lender_ep && nt_donor > 0)
                                   ? (double)nr_donor / nt_donor : 0.0;
            double lender_ho_blk = (cs.is_lender_ep && ht > 0)
                                   ? (double)cs.ho_rej_lender_ep / ht : 0.0;

            double sl1_ho_rate = 0.0, sl1_nc_rate = 0.0;
            double sl0_ho_rate = 0.0, sl0_nc_rate = 0.0;
            {
                int h1 = cs.sl_ho_acc_ep[1] + cs.sl_ho_rej_ep[1];
                int n1 = cs.sl_nc_acc_ep[1] + cs.sl_nc_rej_ep[1];
                int h0 = cs.sl_ho_acc_ep[0] + cs.sl_ho_rej_ep[0];
                int n0 = cs.sl_nc_acc_ep[0] + cs.sl_nc_rej_ep[0];
                sl1_ho_rate = h1>0 ? (double)cs.sl_ho_acc_ep[1]/h1 : 1.0;
                sl1_nc_rate = n1>0 ? (double)cs.sl_nc_acc_ep[1]/n1 : 1.0;
                sl0_ho_rate = h0>0 ? (double)cs.sl_ho_acc_ep[0]/h0 : 1.0;
                sl0_nc_rate = n0>0 ? (double)cs.sl_nc_acc_ep[0]/n0 : 1.0;
            }
            double prio_sat = (
                SLICE_ALPHA[1] * (P_H * sl1_ho_rate + P_N * sl1_nc_rate)
              + SLICE_ALPHA[0] * (P_H * sl0_ho_rate + P_N * sl0_nc_rate)
            ) / PRIO_SAT_DENOM;
            double sf = std::min(cs.sinr_db / 25.0, 1.0);
            o << "\"c" << c << "\":{"
              << "\"ho_acc\":"             << cs.ho_acc_ep
              << ",\"ho_rej\":"            << cs.ho_rej_ep
              << ",\"nc_acc\":"            << cs.nc_acc_ep
              << ",\"nc_rej\":"            << cs.nc_rej_ep
              << ",\"util_served\":"       << util
              << ",\"ho_rate\":"           << ho_rate
              << ",\"nc_rate\":"           << nc_rate
              << ",\"ho_blk\":"            << ho_blk
              << ",\"nc_blk\":"            << nc_blk
              << ",\"lender_ho_blk\":"     << lender_ho_blk
              << ",\"lender_nc_blk\":"     << lender_nc_blk
              << ",\"ho_rej_lend_raw\":"   << cs.ho_rej_lender_ep
              << ",\"nc_rej_lend_raw\":"   << cs.nc_rej_lender_ep
              << ",\"ho_drop_borrower\":"  << ho_drop_borrower
              << ",\"nc_blk_lender\":"     << nc_blk_lender
              << ",\"teleop_ho_rate\":"    << sl1_ho_rate
              << ",\"teleop_nc_rate\":"    << sl1_nc_rate
              << ",\"auto_ho_rate\":"      << sl0_ho_rate
              << ",\"auto_nc_rate\":"      << sl0_nc_rate
              << ",\"prio_sat\":"          << prio_sat
              << ",\"sf_covered\":"        << sf
              << ",\"is_borrower\":"       << (cs.is_borrower_ep ? "true":"false")
              << ",\"is_lender\":"         << (cs.is_lender_ep   ? "true":"false")
              << ",\"ovl\":"               << (cs.overloaded      ? "true":"false")
              << ",\"triggered\":"         << (cs.is_borrower     ? "true":"false")
              << ",\"active_flows\":"      << cs.active_flows.size()
              << ",\"nc_budget_frac\":"    << cs.nc_budget_frac;
            for (uint32_t s = 0; s < N_SL; s++)
                o << ",\"don" << s << "\":" << cs.donation_fracs[s];
            o << "}";
            if (c < N_CELLS - 1) o << ",";
        }
        o << "}";
        return o.str();
    }

    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override {
    auto box = DynamicCast<OpenGymBoxContainer<float>>(action);
    if (!box) return false;

    g_curTick++;

    double saved_used[N_CELLS][N_SL];
    for (uint32_t c = 0; c < N_CELLS; c++) {
        g_cells[c].RecomputeUsed();
        for (uint32_t s = 0; s < N_SL; s++)
            saved_used[c][s] = g_cells[c].used[s];
    }

    for (uint32_t c = 0; c < N_CELLS; c++) {
        g_cells[c].is_lender_prev   = g_cells[c].is_lender;
        g_cells[c].is_borrower_prev = g_cells[c].is_borrower;
    }

    for (uint32_t c = 0; c < N_CELLS; c++) {
        for (uint32_t s = 0; s < N_SL; s++)
            g_cells[c].alloc[s] = CAPS[s];
        g_cells[c].is_lender   = false;
        g_cells[c].is_borrower = false;
        for (uint32_t s = 0; s < N_SL; s++)
            g_cells[c].donation_fracs[s] = 0.0;
    }

    for (uint32_t c = 0; c < N_CELLS; c++) {
        Cell &cs = g_cells[c];

        int ovl_sl = -1;
        double worst_ratio = 0.0;
        for (uint32_t s = 0; s < N_SL; s++) {
            if (!IS_PRIORITY_SLICE[s]) continue;
            double ratio = saved_used[c][s] / (CAPS[s] + 1e-8);
            if (ratio > worst_ratio) { worst_ratio = ratio; ovl_sl = (int)s; }
        }
        if (ovl_sl < 0) ovl_sl = 1;

        double total_borrowed = 0.0;
        static const int DONOR_ORDER[4] = {2, 3, 0, 1};
        for (int di = 0; di < 4; di++) {
            int s = DONOR_ORDER[di];
            if (s == ovl_sl) continue;

            float w = box->GetValue(c * 4 + s);
            if (IS_PRIORITY_SLICE[s] && w < 0.25f) continue;

            double floor_frac;
            double safety_margin;
            if (IS_PRIORITY_SLICE[s]) {
                floor_frac    = 0.62;
                safety_margin = 0.010;
            } else {
                floor_frac    = 0.90;
                safety_margin = 0.025;
            }
            double floor_bw = std::max(CAPS[s] * floor_frac,
                                       saved_used[c][s] + safety_margin);
            double spare = CAPS[s] - floor_bw;
            if (spare < 0.002) continue;

            double max_give_frac;
            if (IS_PRIORITY_SLICE[s])
                max_give_frac = 0.05 + (double)w * 0.20;
            else
                max_give_frac = 0.03 + (double)w * 0.10;
            double give = std::min(spare, CAPS[s] * max_give_frac);
            if (give < 1e-8) continue;

            give = std::min(give, CAPS[ovl_sl] * 1.5 - cs.alloc[ovl_sl]);
            if (give <= 1e-8) continue;

            cs.alloc[s]      -= give;
            cs.alloc[ovl_sl] += give;
            total_borrowed   += give;
            cs.donation_fracs[s] = give / (CAPS[s] + 1e-8);
        }

        bool donated_from_nonpriority = false;
        for (int di = 0; di < 4; di++) {
            int s = DONOR_ORDER[di];
            if (!IS_PRIORITY_SLICE[s] && cs.donation_fracs[s] > 1e-8) {
                donated_from_nonpriority = true;
                break;
            }
        }
        if (donated_from_nonpriority) {
            cs.is_lender    = true;
            cs.is_lender_ep = true;
        }
        cs.is_borrower    = true;
        cs.is_borrower_ep = true;
    }

    for (uint32_t c = 0; c < N_CELLS; c++) {
        Cell &cs = g_cells[c];
        for (int donor : {3, 2}) {
            double hard_min = (donor == 2) ? ALLOC_MIN[2] : ALLOC_MIN[3];
            double floor_bw = std::max(hard_min,
                                       saved_used[c][donor] + 0.020);
            double transferable = cs.alloc[donor] - floor_bw;
            if (transferable < 0.005) continue;

            double need0 = std::max(0.0, saved_used[c][0] - cs.alloc[0]);
            double need1 = std::max(0.0, saved_used[c][1] - cs.alloc[1]);
            double total_need = need0 + need1;

            bool lnc_safe = (cs.lend_nc_blk_ema < LEND_NC_TARGET);
            double frac_idle = lnc_safe ? 0.22 : 0.16;
            double frac_need = lnc_safe ? 0.26 : 0.18;

            if (total_need < 0.005) {
                double r0 = saved_used[c][0] / (cs.alloc[0] + 1e-8);
                double r1 = saved_used[c][1] / (cs.alloc[1] + 1e-8);
                if (r0 < 0.88 && r1 < 0.88) continue;
                double give = std::min(transferable, CAPS[donor] * frac_idle);
                if (r1 >= r0) { cs.alloc[donor] -= give; cs.alloc[1] += give; }
                else          { cs.alloc[donor] -= give; cs.alloc[0] += give; }
            } else {
                double give  = std::min(transferable,
                               std::min(total_need, CAPS[donor] * frac_need));
                double give0 = give * (need0 / total_need);
                double give1 = give * (need1 / total_need);
                cs.alloc[donor] -= (give0 + give1);
                cs.alloc[0]     += give0;
                cs.alloc[1]     += give1;
            }
        }

        double sum = 0.0;
        for (uint32_t s = 0; s < N_SL; s++) sum += cs.alloc[s];
        if (sum > 1.0 + 1e-4)
            for (uint32_t s = 0; s < N_SL; s++) cs.alloc[s] /= sum;
    }

    for (uint32_t c = 0; c < N_CELLS; c++) {
        Cell &cs = g_cells[c];
        cs.RecomputeUsed();
        for (int lp : {2, 3}) {
            double lp_min  = (lp == 2) ? ALLOC_MIN[2] : ALLOC_MIN[3];
            double idle_lp = cs.alloc[lp] - std::max(lp_min,
                                                     saved_used[c][lp] + 0.020);
            if (idle_lp < 0.005) continue;

            bool lnc_safe6 = (cs.lend_nc_blk_ema < LEND_NC_TARGET);
            double give_frac = lnc_safe6 ? 0.65 : 0.50;
            double cap_frac  = lnc_safe6 ? 0.45 : 0.35;
            double give = std::min(idle_lp * give_frac, CAPS[lp] * cap_frac);

            double need0 = std::max(0.0, saved_used[c][0] - cs.alloc[0]);
            double need1 = std::max(0.0, saved_used[c][1] - cs.alloc[1]);
            double tot_need = need0 + need1;
            double r0_p = saved_used[c][0] / (cs.alloc[0] + 1e-8);
            double r1_p = saved_used[c][1] / (cs.alloc[1] + 1e-8);
            if (tot_need < 0.005 && r0_p < 0.88 && r1_p < 0.88) continue;
            if (tot_need < 0.005) {
                if (r1_p >= r0_p) { cs.alloc[lp] -= give; cs.alloc[1] += give; }
                else              { cs.alloc[lp] -= give; cs.alloc[0] += give; }
                continue;
            }
            double g0 = give * (need0 / tot_need);
            double g1 = give * (need1 / tot_need);
            cs.alloc[lp] -= (g0 + g1);
            cs.alloc[0]  += g0;
            cs.alloc[1]  += g1;
        }
        double sum = 0.0;
        for (uint32_t s = 0; s < N_SL; s++) sum += cs.alloc[s];
        if (sum > 1.0 + 1e-4)
            for (uint32_t s = 0; s < N_SL; s++) cs.alloc[s] /= sum;
    }

    for (uint32_t c = 0; c < N_CELLS; c++) {
        Cell &cs = g_cells[c];
        bool clamped = false;
        for (uint32_t s = 0; s < N_SL; s++) {
            double before = cs.alloc[s];
            cs.alloc[s] = std::max(ALLOC_MIN[s],
                                   std::min(ALLOC_MAX[s], cs.alloc[s]));
            if (cs.alloc[s] != before) clamped = true;
        }
        if (clamped) {
            double sum = 0.0;
            for (uint32_t s = 0; s < N_SL; s++) sum += cs.alloc[s];
            for (uint32_t s = 0; s < N_SL; s++) cs.alloc[s] /= sum;
        }
    }

    if (g_curTick < g_simTime)
        LoadTick(g_curEp, g_curTick);

    return true;
}
};

NS_OBJECT_ENSURE_REGISTERED(GnbEnv);
static Ptr<OpenGymInterface> g_iface = nullptr;
static Ptr<GnbEnv>           g_env   = nullptr;

static void GymStep() {
    if (g_curTick >= g_simTime) return;
    g_iface->NotifyCurrentState();
    Simulator::Schedule(Seconds(1.0), &GymStep);
}

int main(int argc, char *argv[]) {
    CommandLine cmd;
    cmd.AddValue("simTime",       "Steps per episode",     g_simTime);
    cmd.AddValue("openGymPort",   "ns3-gym REP port",      g_gymPort);
    cmd.AddValue("tickDir",       "Offline tick dir",      g_tickDir);
    cmd.AddValue("episode",       "Episode index",         g_curEp);
    cmd.AddValue("zmqPull",       "1=ZMQ online, 0=file",  g_zmqPull);
    cmd.AddValue("arrivalPort",   "ZMQ arrival PULL port", g_arrivalPort);
    cmd.AddValue("bridgeHost",    "sumo_bridge host",      g_bridgeHost);
    cmd.AddValue("lambdaArrival", "Vehicle arrival rate",  g_lambdaArrival);
    cmd.Parse(argc, argv);

    NS_LOG_UNCOND("GnbScheduler ep=" << g_curEp
                  << " simTime=" << g_simTime
                  << " port="    << g_gymPort
                  << " zmq="     << (g_zmqPull ? "ON" : "OFF")
                  << " tickDir=" << g_tickDir
                  << " lambda="  << g_lambdaArrival);

    if (g_zmqPull) ZmqInit();
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    for (uint32_t c = 0; c < N_CELLS; c++) {
        g_cells[c].ho_acc_ep = g_cells[c].ho_rej_ep =
        g_cells[c].nc_acc_ep = g_cells[c].nc_rej_ep = 0;
        g_cells[c].ho_rej_lender_ep = g_cells[c].nc_rej_lender_ep = 0;
        g_cells[c].util_served_sum = 0.0;
        g_cells[c].util_ticks      = 0;
        g_cells[c].is_lender_ep    = false;
        g_cells[c].is_borrower_ep  = false;
        g_cells[c].is_lender_prev   = false;
        g_cells[c].is_borrower_prev = false;
        g_cells[c].nc_budget_frac  = NC_BUDGET_BASE;
        g_cells[c].ho_blk_ema = 0.0;
        g_cells[c].nc_blk_ema = 0.0;

        g_cells[c].lagrange_lend_nc = LAGRANGE_INIT;
        g_cells[c].lagrange_lend_ho = LAGRANGE_INIT;
        g_cells[c].lagrange_borr_ho = LAGRANGE_INIT;
        g_cells[c].lend_nc_blk_ema = 0.0;
        g_cells[c].lend_ho_blk_ema = 0.0;
        g_cells[c].borr_ho_blk_ema = 0.0;

        for (uint32_t s = 0; s < N_SL; s++) {
            g_cells[c].sl_ho_acc_ep[s] = g_cells[c].sl_ho_rej_ep[s] = 0;
            g_cells[c].sl_nc_acc_ep[s] = g_cells[c].sl_nc_rej_ep[s] = 0;
            g_cells[c].donation_fracs[s] = 0.0;
        }
    }

    LoadTick(g_curEp, 0);
    PrewarmFlows();

    g_iface = CreateObject<OpenGymInterface>(g_gymPort);
    g_env   = CreateObject<GnbEnv>();
    g_env->SetOpenGymInterface(g_iface);

    Simulator::Schedule(Seconds(0.0), &GymStep);
    Simulator::Stop(Seconds((double)g_simTime + 1.0));
    Simulator::Run();
    Simulator::Destroy();
    g_iface->NotifySimulationEnd();
    if (g_zmqPull) ZmqTeardown();
    return 0;
}