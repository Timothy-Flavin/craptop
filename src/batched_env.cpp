#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <omp.h>
#include <fstream>
#include <string>

namespace py = pybind11;

constexpr int MAP_SIZE = 32;
constexpr int N_AGENTS = 4;
constexpr int VIEW_RANGE = 3;
constexpr float SPEED = 0.5f;
constexpr float DANGER_PENALTY_FACTOR = 0.8f;
constexpr int FLAT_MAP_SIZE = MAP_SIZE * MAP_SIZE;
constexpr float RECENCY_DECAY = 0.99f;
constexpr float MAP_MAX = MAP_SIZE - 0.01f;
constexpr float DIST_SQ_MIN = 0.001f;

// Feature type enum
enum FeatureType {
    EXPECTED_DANGER_FEATURE = 0,
    ACTUAL_DANGER_FEATURE = 1,
    OBSERVED_DANGER_FEATURE = 2,
    OBS_FEATURE = 3,
    EXPECTED_OBS_FEATURE = 4,
    GLOBAL_DISCOVERED_FEATURE = 5,
    OTHER_AGENTS_FEATURE = 6,
    OTHER_AGENTS_LAST_KNOWN_FEATURE = 7,
    GLOBAL_UNDISCOVERED_FEATURE = 8,
    OBS_UNDISCOVERED_FEATURE = 9,
    EXPECTED_OBS_UNDISCOVERED_FEATURE = 10,
    RECENCY_FEATURE = 11,
    RECENCY_STALE_FEATURE = 12,
    WALL_REPEL_FEATURE = 13,
    WALL_ATTRACT_FEATURE = 14
};

// Total floats per environment
constexpr int ENV_STRIDE =
    FLAT_MAP_SIZE +              // expected_danger (0)
    FLAT_MAP_SIZE +              // actual_danger
    (N_AGENTS * FLAT_MAP_SIZE) + // observed_danger
    (N_AGENTS * FLAT_MAP_SIZE) + // obs (mask)
    (N_AGENTS * 2) +             // agent_locations
    (N_AGENTS * FLAT_MAP_SIZE) + // expected_obs
    (N_AGENTS * 2 * N_AGENTS) +  // last_agent_locations
    FLAT_MAP_SIZE +              // global_discovered
    (N_AGENTS * FLAT_MAP_SIZE);  // recency

struct GameStateView {
    float* expected_danger;
    float* actual_danger;
    float* observed_danger;
    float* obs;
    float* agent_locations;
    float* expected_obs;
    float* last_agent_locations;
    float* global_discovered;
    float* recency;
};

// ─── Force helpers ──────────────────────────────────────────────────────────

// Force contribution per tile: mass * direction / dist^(pow+1)
// i.e. out += dx * mass / denom, out += dy * mass / denom
// where denom = dist^(pow+1).  For pow=1: dist^2, pow=2: dist^3.
// We compute denom inline via switch to avoid function-call overhead
// and divide directly (the result `f` is reused for both dx and dy components).

// Gravity from a map (global). Iterates all MAP_SIZE*MAP_SIZE tiles.
void get_gravity(const float* __restrict map, int pow,
                 float agent_x, float agent_y,
                 float& out_gx, float& out_gy, bool invert) {
    float gx = 0.0f, gy = 0.0f;

    for (int y = 0; y < MAP_SIZE; ++y) {
        const float dy = static_cast<float>(y) - agent_y;
        const float* row = map + y * MAP_SIZE;
        for (int x = 0; x < MAP_SIZE; ++x) {
            float mass = invert ? (1.0f - row[x]) : row[x];
            if (mass <= 0.001f) continue;

            const float dx = static_cast<float>(x) - agent_x;
            const float dist_sq = dx * dx + dy * dy;
            if (dist_sq < DIST_SQ_MIN) continue;

            const float dist = std::sqrt(dist_sq);
            float denom;
            switch (pow) {
                case 1: denom = dist_sq; break;
                case 2: denom = dist * dist_sq; break;
                default: { denom = dist; for (int p = 0; p < pow; ++p) denom *= dist; break; }
            }
            const float f = mass / denom;

            gx += dx * f;
            gy += dy * f;
        }
    }
    out_gx = gx;
    out_gy = gy;
}

// Gravity from a map (local) – only tiles within VIEW_RANGE.
void get_gravity_local(const float* __restrict map, int pow,
                       float agent_x, float agent_y,
                       float& out_gx, float& out_gy, bool invert) {
    float gx = 0.0f, gy = 0.0f;

    const int yc = static_cast<int>(agent_y);
    const int xc = static_cast<int>(agent_x);
    const int y_s = std::max(0, yc - VIEW_RANGE);
    const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
    const int x_s = std::max(0, xc - VIEW_RANGE);
    const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

    for (int y = y_s; y < y_e; ++y) {
        const float dy = static_cast<float>(y) - agent_y;
        const float* row = map + y * MAP_SIZE;
        for (int x = x_s; x < x_e; ++x) {
            float mass = invert ? (1.0f - row[x]) : row[x];
            if (mass <= 0.001f) continue;

            const float dx = static_cast<float>(x) - agent_x;
            const float dist_sq = dx * dx + dy * dy;
            if (dist_sq < DIST_SQ_MIN) continue;

            const float dist = std::sqrt(dist_sq);
            float denom;
            switch (pow) {
                case 1: denom = dist_sq; break;
                case 2: denom = dist * dist_sq; break;
                default: { denom = dist; for (int p = 0; p < pow; ++p) denom *= dist; break; }
            }
            const float f = mass / denom;

            gx += dx * f;
            gy += dy * f;
        }
    }
    out_gx = gx;
    out_gy = gy;
}

// Gravity from other agents (global – all agents considered).
void get_gravity_from_agents(const float* __restrict all_locs, int skip_idx, int pow,
                             float agent_x, float agent_y,
                             float& out_gx, float& out_gy) {
    float gx = 0.0f, gy = 0.0f;

    for (int j = 0; j < N_AGENTS; ++j) {
        if (j == skip_idx) continue;
        const float other_y = all_locs[j * 2];
        const float other_x = all_locs[j * 2 + 1];
        const float dx = other_x - agent_x;
        const float dy = other_y - agent_y;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq < DIST_SQ_MIN) continue;
        const float dist = std::sqrt(dist_sq);
        float denom;
        switch (pow) {
            case 1: denom = dist_sq; break;
            case 2: denom = dist * dist_sq; break;
            default: { denom = dist; for (int p = 0; p < pow; ++p) denom *= dist; break; }
        }
        const float f = 1.0f / denom;
        gx += dx * f;
        gy += dy * f;
    }
    out_gx = gx;
    out_gy = gy;
}

// Gravity from other agents (local – only within VIEW_RANGE).
void get_gravity_from_agents_local(const float* __restrict all_locs, int skip_idx, int pow,
                                   float agent_x, float agent_y,
                                   float& out_gx, float& out_gy) {
    float gx = 0.0f, gy = 0.0f;
    const int yc = static_cast<int>(agent_y);
    const int xc = static_cast<int>(agent_x);

    for (int j = 0; j < N_AGENTS; ++j) {
        if (j == skip_idx) continue;
        const float other_y = all_locs[j * 2];
        const float other_x = all_locs[j * 2 + 1];
        if (std::abs(static_cast<int>(other_y) - yc) > VIEW_RANGE ||
            std::abs(static_cast<int>(other_x) - xc) > VIEW_RANGE) continue;

        const float dx = other_x - agent_x;
        const float dy = other_y - agent_y;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq < DIST_SQ_MIN) continue;
        const float dist = std::sqrt(dist_sq);
        float denom;
        switch (pow) {
            case 1: denom = dist_sq; break;
            case 2: denom = dist * dist_sq; break;
            default: { denom = dist; for (int p = 0; p < pow; ++p) denom *= dist; break; }
        }
        const float f = 1.0f / denom;
        gx += dx * f;
        gy += dy * f;
    }
    out_gx = gx;
    out_gy = gy;
}

// Wall border force: virtual tiles of mass=1.0 at x=-1, x=MAP_SIZE, y=-1, y=MAP_SIZE
// along each edge. When local=true, only wall tiles within VIEW_RANGE are considered.
void get_gravity_wall(int pow, float agent_x, float agent_y,
                      float& out_gx, float& out_gy, bool local) {
    float gx = 0.0f, gy = 0.0f;

    // Determine iteration bounds
    int start, end;
    if (local) {
        const int yc = static_cast<int>(agent_y);
        const int xc = static_cast<int>(agent_x);
        // We iterate the wall coordinate range that falls within view
        // Wall tiles are at -1 and MAP_SIZE, so check if they're within VIEW_RANGE
        // For the top/bottom/left/right walls, we iterate tile positions along the wall
        start = -1; // walls are always at -1 and MAP_SIZE
        end = MAP_SIZE + 1;
    } else {
        start = -1;
        end = MAP_SIZE + 1;
    }

    auto accumulate_tile = [&](float wx, float wy) {
        const float dx = wx - agent_x;
        const float dy = wy - agent_y;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq < DIST_SQ_MIN) return;
        const float dist = std::sqrt(dist_sq);
        float denom;
        switch (pow) {
            case 1: denom = dist_sq; break;
            case 2: denom = dist * dist_sq; break;
            default: { denom = dist; for (int p = 0; p < pow; ++p) denom *= dist; break; }
        }
        const float f = 1.0f / denom;
        gx += dx * f;
        gy += dy * f;
    };

    if (local) {
        const int yc = static_cast<int>(agent_y);
        const int xc = static_cast<int>(agent_x);

        // Top wall (y = -1): only if within view range vertically
        if (yc - (-1) <= VIEW_RANGE) {
            const int xs = std::max(-1, xc - VIEW_RANGE);
            const int xe = std::min(MAP_SIZE, xc + VIEW_RANGE) + 1;
            for (int x = xs; x < xe; ++x)
                accumulate_tile(static_cast<float>(x), -1.0f);
        }
        // Bottom wall (y = MAP_SIZE)
        if (MAP_SIZE - yc <= VIEW_RANGE) {
            const int xs = std::max(-1, xc - VIEW_RANGE);
            const int xe = std::min(MAP_SIZE, xc + VIEW_RANGE) + 1;
            for (int x = xs; x < xe; ++x)
                accumulate_tile(static_cast<float>(x), static_cast<float>(MAP_SIZE));
        }
        // Left wall (x = -1)
        if (xc - (-1) <= VIEW_RANGE) {
            const int ys = std::max(-1, yc - VIEW_RANGE);
            const int ye = std::min(MAP_SIZE, yc + VIEW_RANGE) + 1;
            for (int y = ys; y < ye; ++y)
                accumulate_tile(-1.0f, static_cast<float>(y));
        }
        // Right wall (x = MAP_SIZE)
        if (MAP_SIZE - xc <= VIEW_RANGE) {
            const int ys = std::max(-1, yc - VIEW_RANGE);
            const int ye = std::min(MAP_SIZE, yc + VIEW_RANGE) + 1;
            for (int y = ys; y < ye; ++y)
                accumulate_tile(static_cast<float>(MAP_SIZE), static_cast<float>(y));
        }
    } else {
        // Global: all wall tiles
        for (int i = -1; i <= MAP_SIZE; ++i) {
            const float fi = static_cast<float>(i);
            accumulate_tile(fi, -1.0f);                             // top wall
            accumulate_tile(fi, static_cast<float>(MAP_SIZE));      // bottom wall
            accumulate_tile(-1.0f, fi);                             // left wall
            accumulate_tile(static_cast<float>(MAP_SIZE), fi);      // right wall
        }
    }

    out_gx = gx;
    out_gy = gy;
}

// ─── BatchedEnvironment ─────────────────────────────────────────────────────

class BatchedEnvironment {
public:
    int num_envs;
    int seed;
    std::vector<std::mt19937> rngs;
    std::vector<float> data;
    std::vector<std::string> map_paths;
    std::vector<std::string> expected_map_paths;
    std::vector<bool> env_terminated;

    BatchedEnvironment(int n_envs, int sim_seed,
                       std::vector<std::string> maps = {},
                       std::vector<std::string> expected_maps = {})
        : num_envs(n_envs), seed(sim_seed), map_paths(std::move(maps)),
          expected_map_paths(std::move(expected_maps))
    {
        data.resize(static_cast<size_t>(num_envs) * ENV_STRIDE, 0.0f);
        rngs.resize(num_envs);
        env_terminated.assign(num_envs, false);
        for (int i = 0; i < num_envs; ++i)
            rngs[i].seed(seed + i);
        reset();
    }

    void bind_state(GameStateView& s, int env_idx) {
        float* ptr = data.data() + (static_cast<size_t>(env_idx) * ENV_STRIDE);
        s.expected_danger     = ptr; ptr += FLAT_MAP_SIZE;
        s.actual_danger       = ptr; ptr += FLAT_MAP_SIZE;
        s.observed_danger     = ptr; ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.obs                 = ptr; ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.agent_locations     = ptr; ptr += N_AGENTS * 2;
        s.expected_obs        = ptr; ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.last_agent_locations = ptr; ptr += N_AGENTS * 2 * N_AGENTS;
        s.global_discovered   = ptr; ptr += FLAT_MAP_SIZE;
        s.recency             = ptr;
    }

    void reset_env(GameStateView& s, int e) {
        std::memset(data.data() + (static_cast<size_t>(e) * ENV_STRIDE), 0,
                    ENV_STRIDE * sizeof(float));

        // Actual danger
        if (!map_paths.empty()) {
            const std::string& path = map_paths[e % map_paths.size()];
            std::ifstream file(path, std::ios::binary);
            if (file.is_open()) {
                file.read(reinterpret_cast<char*>(s.actual_danger), FLAT_MAP_SIZE * sizeof(float));
            } else {
                generate_procedural_danger(s.actual_danger, e);
            }
        } else {
            generate_procedural_danger(s.actual_danger, e);
        }

        // Expected danger
        if (!expected_map_paths.empty()) {
            const std::string& path = expected_map_paths[e % expected_map_paths.size()];
            std::ifstream file(path, std::ios::binary);
            if (file.is_open()) {
                file.read(reinterpret_cast<char*>(s.expected_danger), FLAT_MAP_SIZE * sizeof(float));
            }
            // else: already zeroed by memset
        }
        // else: already zeroed

        // Agent locations: center of map
        const float center = MAP_SIZE * 0.5f;
        for (int i = 0; i < N_AGENTS; ++i) {
            s.agent_locations[i * 2]     = center;
            s.agent_locations[i * 2 + 1] = center;
        }

        // Init observed_danger from expected_danger
        for (int i = 0; i < N_AGENTS; ++i)
            std::memcpy(s.observed_danger + i * FLAT_MAP_SIZE,
                        s.expected_danger, FLAT_MAP_SIZE * sizeof(float));

        update_obs(s);
    }

    void reset() {
        env_terminated.assign(num_envs, false);
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e) {
            GameStateView s;
            bind_state(s, e);
            reset_env(s, e);
        }
    }

    std::pair<py::array_t<float>, py::array_t<bool>> step(
            py::array_t<float> actions_array, float communication_prob = -1.0f) {
        auto r = actions_array.unchecked<2>();
        py::array_t<float> rewards_array({num_envs, N_AGENTS});
        auto rewards_ptr = rewards_array.mutable_unchecked<2>();
        py::array_t<bool> terminated_array({num_envs});
        auto terminated_ptr = terminated_array.mutable_unchecked<1>();

        #pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e) {
            GameStateView s;
            bind_state(s, e);

            if (env_terminated[e]) {
                reset_env(s, e);
                env_terminated[e] = false;
            }

            update_locations(s, r, e);
            update_obs(s);
            update_recency(s);
            update_last_location(s, e, communication_prob);
            bool done = calc_rewards(s, rewards_ptr, e);
            terminated_ptr(e) = done;
            env_terminated[e] = done;
        }
        return {rewards_array, terminated_array};
    }

    std::pair<size_t, size_t> get_memory_view() {
        return {reinterpret_cast<size_t>(data.data()), data.size() * sizeof(float)};
    }

    int get_stride() const { return ENV_STRIDE; }
    int get_flat_map_size() const { return FLAT_MAP_SIZE; }

    py::array_t<float> get_gravity_attractions(
            py::object agent_mask_obj,
            int feature_type,
            int pow = 2,
            bool normalize = false,
            bool local = false)
    {
        // Parse agent mask
        bool agent_mask[N_AGENTS];
        if (agent_mask_obj.is_none()) {
            for (int i = 0; i < N_AGENTS; ++i) agent_mask[i] = true;
        } else {
            auto mask = agent_mask_obj.cast<py::array_t<bool>>();
            auto mask_ptr = mask.data();
            for (int i = 0; i < N_AGENTS; ++i) agent_mask[i] = mask_ptr[i];
        }

        py::array_t<float> output_array({num_envs, N_AGENTS, 2});
        auto output_ptr = output_array.mutable_unchecked<3>();

        #pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e) {
            GameStateView s;
            bind_state(s, e);

            for (int i = 0; i < N_AGENTS; ++i) {
                if (!agent_mask[i]) {
                    output_ptr(e, i, 0) = 0.0f;
                    output_ptr(e, i, 1) = 0.0f;
                    continue;
                }

                const float agent_y = s.agent_locations[i * 2];
                const float agent_x = s.agent_locations[i * 2 + 1];
                float gx = 0.0f, gy = 0.0f;

                switch (static_cast<FeatureType>(feature_type)) {
                // ── Map-based features (no invert) ──
                case EXPECTED_DANGER_FEATURE:
                    dispatch_map_gravity(s.expected_danger, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case ACTUAL_DANGER_FEATURE:
                    dispatch_map_gravity(s.actual_danger, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case OBSERVED_DANGER_FEATURE:
                    dispatch_map_gravity(s.observed_danger + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case OBS_FEATURE:
                    dispatch_map_gravity(s.obs + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case EXPECTED_OBS_FEATURE:
                    dispatch_map_gravity(s.expected_obs + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case GLOBAL_DISCOVERED_FEATURE:
                    dispatch_map_gravity(s.global_discovered, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case RECENCY_FEATURE:
                    dispatch_map_gravity(s.recency + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, false, local);
                    break;

                // ── Map-based features (inverted) ──
                case GLOBAL_UNDISCOVERED_FEATURE:
                    dispatch_map_gravity(s.global_discovered, pow, agent_x, agent_y, gx, gy, true, local);
                    break;
                case OBS_UNDISCOVERED_FEATURE:
                    dispatch_map_gravity(s.obs + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, true, local);
                    break;
                case EXPECTED_OBS_UNDISCOVERED_FEATURE:
                    dispatch_map_gravity(s.expected_obs + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, true, local);
                    break;
                case RECENCY_STALE_FEATURE:
                    dispatch_map_gravity(s.recency + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, true, local);
                    break;

                // ── Agent-based features ──
                case OTHER_AGENTS_FEATURE:
                    if (local)
                        get_gravity_from_agents_local(s.agent_locations, i, pow, agent_x, agent_y, gx, gy);
                    else
                        get_gravity_from_agents(s.agent_locations, i, pow, agent_x, agent_y, gx, gy);
                    break;
                case OTHER_AGENTS_LAST_KNOWN_FEATURE: {
                    float* last_known = s.last_agent_locations + i * 2 * N_AGENTS;
                    if (local)
                        get_gravity_from_agents_local(last_known, i, pow, agent_x, agent_y, gx, gy);
                    else
                        get_gravity_from_agents(last_known, i, pow, agent_x, agent_y, gx, gy);
                    break;
                }

                // ── Wall features ──
                case WALL_REPEL_FEATURE:
                    get_gravity_wall(pow, agent_x, agent_y, gx, gy, local);
                    // Negate so it repels (wall tiles attract by default; flip to repel)
                    gx = -gx;
                    gy = -gy;
                    break;
                case WALL_ATTRACT_FEATURE:
                    get_gravity_wall(pow, agent_x, agent_y, gx, gy, local);
                    break;

                default:
                    gx = gy = 0.0f;
                    break;
                }

                // Normalize
                if (normalize) {
                    const float mag = std::sqrt(gx * gx + gy * gy);
                    if (mag > 1.0f) {
                        const float inv_mag = 1.0f / mag;
                        gx *= inv_mag;
                        gy *= inv_mag;
                    }
                }

                output_ptr(e, i, 0) = gy; // dy
                output_ptr(e, i, 1) = gx; // dx
            }
        }
        return output_array;
    }

private:
    static void generate_procedural_danger(float* danger, int e) {
        const float ef = static_cast<float>(e);
        for (int i = 0; i < FLAT_MAP_SIZE; ++i) {
            const int y = i / MAP_SIZE;
            const int x = i % MAP_SIZE;
            float val = (std::sin(static_cast<float>(x) * 0.3f + ef) +
                         std::cos(static_cast<float>(y) * 0.3f + ef * 2.0f)) * 0.5f;
            danger[i] = std::fmin(1.0f, std::fmax(-1.0f, val));
        }
    }

    static void dispatch_map_gravity(const float* map, int pow,
                                     float agent_x, float agent_y,
                                     float& gx, float& gy,
                                     bool invert, bool local) {
        if (local)
            get_gravity_local(map, pow, agent_x, agent_y, gx, gy, invert);
        else
            get_gravity(map, pow, agent_x, agent_y, gx, gy, invert);
    }

    void update_locations(GameStateView& s, const py::detail::unchecked_reference<float, 2>& actions, int env_idx) {
        for (int i = 0; i < N_AGENTS; ++i) {
            float dy = actions(env_idx, i * 2);
            float dx = actions(env_idx, i * 2 + 1);

            // Normalize direction
            const float len_sq = dy * dy + dx * dx;
            if (len_sq > 0.00000001f) {
                const float inv_len = 1.0f / std::sqrt(len_sq);
                dy *= inv_len;
                dx *= inv_len;
            }

            const int cy = std::clamp(static_cast<int>(s.agent_locations[i * 2]), 0, MAP_SIZE - 1);
            const int cx = std::clamp(static_cast<int>(s.agent_locations[i * 2 + 1]), 0, MAP_SIZE - 1);

            const float danger = s.actual_danger[cy * MAP_SIZE + cx];
            const float effective_speed = SPEED * (1.0f - danger * DANGER_PENALTY_FACTOR);

            const float ny = s.agent_locations[i * 2]     + dy * effective_speed;
            const float nx = s.agent_locations[i * 2 + 1] + dx * effective_speed;

            s.agent_locations[i * 2]     = std::clamp(ny, 0.0f, MAP_MAX);
            s.agent_locations[i * 2 + 1] = std::clamp(nx, 0.0f, MAP_MAX);
        }
    }

    static void update_obs(GameStateView& s) {
        for (int i = 0; i < N_AGENTS; ++i) {
            const int yc = static_cast<int>(s.agent_locations[i * 2]);
            const int xc = static_cast<int>(s.agent_locations[i * 2 + 1]);
            const int y_s = std::max(0, yc - VIEW_RANGE);
            const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
            const int x_s = std::max(0, xc - VIEW_RANGE);
            const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

            float* obs_i = s.obs + i * FLAT_MAP_SIZE;
            float* od_i  = s.observed_danger + i * FLAT_MAP_SIZE;

            for (int ly = y_s; ly < y_e; ++ly) {
                const int row_off = ly * MAP_SIZE;
                for (int lx = x_s; lx < x_e; ++lx) {
                    const int idx = row_off + lx;
                    obs_i[idx] = 1.0f;
                    od_i[idx]  = s.actual_danger[idx];
                }
            }
        }
    }

    static void update_recency(GameStateView& s) {
        for (int i = 0; i < N_AGENTS; ++i) {
            float* agent_recency = s.recency + i * FLAT_MAP_SIZE;

            // Decay all tiles (contiguous memory, cache-friendly)
            for (int j = 0; j < FLAT_MAP_SIZE; ++j)
                agent_recency[j] *= RECENCY_DECAY;

            // Set tiles within view range to 1.0
            const int yc = static_cast<int>(s.agent_locations[i * 2]);
            const int xc = static_cast<int>(s.agent_locations[i * 2 + 1]);
            const int y_s = std::max(0, yc - VIEW_RANGE);
            const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
            const int x_s = std::max(0, xc - VIEW_RANGE);
            const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

            for (int ly = y_s; ly < y_e; ++ly) {
                // Use memset-like fill for contiguous run of floats in a row
                float* row_start = agent_recency + ly * MAP_SIZE + x_s;
                const int run_len = x_e - x_s;
                for (int k = 0; k < run_len; ++k)
                    row_start[k] = 1.0f;
            }
        }
    }

    void update_last_location(GameStateView& s, int env_idx, float p) {
        const bool try_radio = (p > 0.0f && p <= 1.0f);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < N_AGENTS; ++i) {
            const int viewer_y = static_cast<int>(s.agent_locations[i * 2]);
            const int viewer_x = static_cast<int>(s.agent_locations[i * 2 + 1]);
            float* my_last = s.last_agent_locations + i * (2 * N_AGENTS);

            for (int j = 0; j < N_AGENTS; ++j) {
                if (i == j) {
                    my_last[j * 2]     = s.agent_locations[j * 2];
                    my_last[j * 2 + 1] = s.agent_locations[j * 2 + 1];
                    continue;
                }

                const int target_y = static_cast<int>(s.agent_locations[j * 2]);
                const int target_x = static_cast<int>(s.agent_locations[j * 2 + 1]);

                bool updated = (std::abs(viewer_y - target_y) <= VIEW_RANGE &&
                                std::abs(viewer_x - target_x) <= VIEW_RANGE);

                if (!updated && try_radio && dist(rngs[env_idx]) < p)
                    updated = true;

                if (updated) {
                    my_last[j * 2]     = s.agent_locations[j * 2];
                    my_last[j * 2 + 1] = s.agent_locations[j * 2 + 1];
                }
            }
        }
    }

    bool calc_rewards(GameStateView& s, py::detail::unchecked_mutable_reference<float, 2>& rewards, int env_idx) {
        for (int i = 0; i < N_AGENTS; ++i)
            rewards(env_idx, i) = 0.0f;

        int undiscovered_count = 0;

        // Pre-fetch integer agent positions once
        int ay[N_AGENTS], ax[N_AGENTS];
        for (int i = 0; i < N_AGENTS; ++i) {
            ay[i] = static_cast<int>(s.agent_locations[i * 2]);
            ax[i] = static_cast<int>(s.agent_locations[i * 2 + 1]);
        }

        for (int y = 0; y < MAP_SIZE; ++y) {
            for (int x = 0; x < MAP_SIZE; ++x) {
                const int idx = y * MAP_SIZE + x;
                if (s.global_discovered[idx] > 0.5f) continue;

                int seeing_count = 0;
                bool seen_by[N_AGENTS] = {};

                for (int i = 0; i < N_AGENTS; ++i) {
                    if (std::abs(ay[i] - y) <= VIEW_RANGE &&
                        std::abs(ax[i] - x) <= VIEW_RANGE) {
                        seen_by[i] = true;
                        ++seeing_count;
                    }
                }

                if (seeing_count > 0) {
                    s.global_discovered[idx] = 1.0f;
                    const float share = 1.0f / static_cast<float>(seeing_count);
                    for (int i = 0; i < N_AGENTS; ++i) {
                        if (seen_by[i]) rewards(env_idx, i) += share;
                    }
                } else {
                    ++undiscovered_count;
                }
            }
        }

        if (undiscovered_count == 0) {
            for (int i = 0; i < N_AGENTS; ++i) rewards(env_idx, i) += 10.0f;
            return true;
        }
        return false;
    }
};

// ─── Pybind11 module ────────────────────────────────────────────────────────

PYBIND11_MODULE(_core, m) {
    py::enum_<FeatureType>(m, "FeatureType")
        .value("EXPECTED_DANGER", EXPECTED_DANGER_FEATURE)
        .value("ACTUAL_DANGER", ACTUAL_DANGER_FEATURE)
        .value("OBSERVED_DANGER", OBSERVED_DANGER_FEATURE)
        .value("OBS", OBS_FEATURE)
        .value("EXPECTED_OBS", EXPECTED_OBS_FEATURE)
        .value("GLOBAL_DISCOVERED", GLOBAL_DISCOVERED_FEATURE)
        .value("OTHER_AGENTS", OTHER_AGENTS_FEATURE)
        .value("OTHER_AGENTS_LAST_KNOWN", OTHER_AGENTS_LAST_KNOWN_FEATURE)
        .value("GLOBAL_UNDISCOVERED", GLOBAL_UNDISCOVERED_FEATURE)
        .value("OBS_UNDISCOVERED", OBS_UNDISCOVERED_FEATURE)
        .value("EXPECTED_OBS_UNDISCOVERED", EXPECTED_OBS_UNDISCOVERED_FEATURE)
        .value("RECENCY", RECENCY_FEATURE)
        .value("RECENCY_STALE", RECENCY_STALE_FEATURE)
        .value("WALL_REPEL", WALL_REPEL_FEATURE)
        .value("WALL_ATTRACT", WALL_ATTRACT_FEATURE);

    py::class_<BatchedEnvironment>(m, "BatchedEnvironment")
        .def(py::init<int, int, std::vector<std::string>, std::vector<std::string>>(),
             py::arg("n_envs"),
             py::arg("seed") = 42,
             py::arg("map_paths") = std::vector<std::string>(),
             py::arg("expected_map_paths") = std::vector<std::string>())
        .def("reset", &BatchedEnvironment::reset)
        .def("step", &BatchedEnvironment::step,
             py::arg("actions"), py::arg("communication_prob") = -1.0f)
        .def("get_memory_view", &BatchedEnvironment::get_memory_view)
        .def("get_stride", &BatchedEnvironment::get_stride)
        .def("get_flat_map_size", &BatchedEnvironment::get_flat_map_size)
        .def("get_gravity_attractions", &BatchedEnvironment::get_gravity_attractions,
             py::arg("agent_mask") = py::none(),
             py::arg("feature_type"),
             py::arg("pow") = 2,
             py::arg("normalize") = false,
             py::arg("local") = false)
        .def_readonly("num_envs", &BatchedEnvironment::num_envs);
}
