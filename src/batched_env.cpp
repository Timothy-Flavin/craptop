// batched_env.cpp - Partial-observability (radio) mode
// Each agent maintains its own obs mask, observed_danger, expected_obs,
// and last_agent_locations.  Radio communication updates positions
// probabilistically each frame.
//
// Exposed as pybind11 module: multi_agent_coverage._core

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstring>
#include <random>
#include <omp.h>
#include <fstream>
#include <string>

#include "gravity.h"

#ifndef ssize_t
#  ifdef _WIN32
     typedef ptrdiff_t ssize_t;
#  endif
#endif

namespace py = pybind11;

// --- State layout (partial-obs) ---

constexpr int ENV_STRIDE =
    FLAT_MAP_SIZE +              // expected_danger      (prior belief, shared)
    FLAT_MAP_SIZE +              // actual_danger         (ground truth, shared)
    (N_AGENTS * FLAT_MAP_SIZE) + // observed_danger       (per-agent danger belief)
    (N_AGENTS * FLAT_MAP_SIZE) + // obs                   (per-agent visibility mask)
    (N_AGENTS * 2) +             // agent_locations       (ground truth positions)
    (N_AGENTS * FLAT_MAP_SIZE) + // expected_obs          (per-agent belief about all agents' obs)
    (N_AGENTS * 2 * N_AGENTS) +  // last_agent_locations  (per-agent belief about others' positions)
    FLAT_MAP_SIZE +              // global_discovered     (env-level, for rewards)
    (N_AGENTS * FLAT_MAP_SIZE);  // recency               (per-agent)

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

// --- BatchedEnvironment (partial-obs) ---

class BatchedEnvironment {
public:
    int num_envs;
    int seed;
    std::vector<std::mt19937> rngs;
    std::vector<float> data;
    std::vector<std::string> map_paths;
    std::vector<std::string> expected_map_paths;
    std::vector<bool> env_terminated;
    std::vector<int> undiscovered_remaining;

    BatchedEnvironment(int n_envs, int sim_seed,
                       std::vector<std::string> maps = {},
                       std::vector<std::string> expected_maps = {})
        : num_envs(n_envs), seed(sim_seed), map_paths(std::move(maps)),
          expected_map_paths(std::move(expected_maps))
    {
        data.resize(static_cast<size_t>(num_envs) * ENV_STRIDE, 0.0f);
        rngs.resize(num_envs);
        env_terminated.assign(num_envs, false);
        undiscovered_remaining.assign(num_envs, FLAT_MAP_SIZE);
        for (int i = 0; i < num_envs; ++i)
            rngs[i].seed(seed + i);
        reset();
    }

    void bind_state(GameStateView& s, int env_idx) {
        float* ptr = data.data() + (static_cast<size_t>(env_idx) * ENV_STRIDE);
        s.expected_danger      = ptr; ptr += FLAT_MAP_SIZE;
        s.actual_danger        = ptr; ptr += FLAT_MAP_SIZE;
        s.observed_danger      = ptr; ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.obs                  = ptr; ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.agent_locations      = ptr; ptr += N_AGENTS * 2;
        s.expected_obs         = ptr; ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.last_agent_locations = ptr; ptr += N_AGENTS * 2 * N_AGENTS;
        s.global_discovered    = ptr; ptr += FLAT_MAP_SIZE;
        s.recency              = ptr;
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

        // Expected danger (prior belief / satellite data)
        if (!expected_map_paths.empty()) {
            const std::string& path = expected_map_paths[e % expected_map_paths.size()];
            std::ifstream file(path, std::ios::binary);
            if (file.is_open()) {
                file.read(reinterpret_cast<char*>(s.expected_danger), FLAT_MAP_SIZE * sizeof(float));
            }
        }

        // Agent locations: center of map
        const float center = MAP_SIZE * 0.5f;
        for (int i = 0; i < N_AGENTS; ++i) {
            s.agent_locations[i * 2]     = center;
            s.agent_locations[i * 2 + 1] = center;
        }

        // Init last_agent_locations to actual positions
        for (int i = 0; i < N_AGENTS; ++i) {
            float* my_last = s.last_agent_locations + i * (2 * N_AGENTS);
            for (int j = 0; j < N_AGENTS; ++j) {
                my_last[j * 2]     = s.agent_locations[j * 2];
                my_last[j * 2 + 1] = s.agent_locations[j * 2 + 1];
            }
        }

        // Init observed_danger from expected_danger
        for (int i = 0; i < N_AGENTS; ++i)
            std::memcpy(s.observed_danger + i * FLAT_MAP_SIZE,
                        s.expected_danger, FLAT_MAP_SIZE * sizeof(float));

        undiscovered_remaining[e] = FLAT_MAP_SIZE;
        update_obs(s);
        update_expected_obs(s);
        // Count tiles discovered on reset
        for (int idx = 0; idx < FLAT_MAP_SIZE; ++idx) {
            if (s.global_discovered[idx] > 0.5f)
                undiscovered_remaining[e]--;
        }
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
            update_expected_obs(s);
            bool done = calc_rewards(s, rewards_ptr, e);
            terminated_ptr(e) = done;
            env_terminated[e] = done;
        }
        return {rewards_array, terminated_array};
    }

    std::pair<size_t, size_t> get_memory_view() {
        return {reinterpret_cast<size_t>(data.data()), data.size() * sizeof(float)};
    }

    // Returns a writable numpy array that is a zero-copy view of the state buffer.
    // Python can read and modify state in-place; no data is copied.
    py::array_t<float> get_state() {
        py::capsule base(data.data(), [](void*) {});  // no-op: memory owned by C++
        return py::array_t<float>(
            {static_cast<ssize_t>(num_envs), static_cast<ssize_t>(ENV_STRIDE)},
            {static_cast<ssize_t>(ENV_STRIDE) * static_cast<ssize_t>(sizeof(float)),
             static_cast<ssize_t>(sizeof(float))},
            data.data(),
            base);
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
                case WALL_REPEL_FEATURE:
                    get_gravity_wall(pow, agent_x, agent_y, gx, gy, local);
                    gx = -gx;
                    gy = -gy;
                    break;
                case WALL_ATTRACT_FEATURE:
                    get_gravity_wall(pow, agent_x, agent_y, gx, gy, local);
                    break;
                case GLOBAL_VORONOI_UNDISCOVERED_FEATURE:
                    if (local)
                        get_gravity_voronoi_local(s.global_discovered, s.agent_locations, i, pow, agent_x, agent_y, gx, gy);
                    else
                        get_gravity_voronoi(s.global_discovered, s.agent_locations, i, pow, agent_x, agent_y, gx, gy);
                    break;
                case EXPECTED_VORONOI_UNDISCOVERED_FEATURE: {
                    // Agent i's belief: expected_obs[i] for discovery, last_agent_locations[i] for positions
                    float* eobs_i = s.expected_obs + i * FLAT_MAP_SIZE;
                    float* last_locs_i = s.last_agent_locations + i * 2 * N_AGENTS;
                    if (local)
                        get_gravity_voronoi_local(eobs_i, last_locs_i, i, pow, agent_x, agent_y, gx, gy);
                    else
                        get_gravity_voronoi(eobs_i, last_locs_i, i, pow, agent_x, agent_y, gx, gy);
                    break;
                }
                default:
                    gx = gy = 0.0f;
                    break;
                }

                if (normalize) {
                    const float mag = std::sqrt(gx * gx + gy * gy);
                    if (mag > 1.0f) {
                        const float inv_mag = 1.0f / mag;
                        gx *= inv_mag;
                        gy *= inv_mag;
                    }
                }

                output_ptr(e, i, 0) = gy;
                output_ptr(e, i, 1) = gx;
            }
        }
        return output_array;
    }

private:
    void update_locations(GameStateView& s,
                          const py::detail::unchecked_reference<float, 2>& actions,
                          int env_idx) {
        for (int i = 0; i < N_AGENTS; ++i) {
            float dy = actions(env_idx, i * 2);
            float dx = actions(env_idx, i * 2 + 1);

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

            s.agent_locations[i * 2]     = std::clamp(s.agent_locations[i * 2]     + dy * effective_speed, 0.0f, MAP_MAX);
            s.agent_locations[i * 2 + 1] = std::clamp(s.agent_locations[i * 2 + 1] + dx * effective_speed, 0.0f, MAP_MAX);
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

    // Agent i's belief about what ALL agents have observed, based on
    // last_agent_locations.  Cumulative (never cleared).
    static void update_expected_obs(GameStateView& s) {
        for (int i = 0; i < N_AGENTS; ++i) {
            float* eobs_i = s.expected_obs + i * FLAT_MAP_SIZE;
            float* my_last = s.last_agent_locations + i * (2 * N_AGENTS);

            for (int j = 0; j < N_AGENTS; ++j) {
                const float loc_y = my_last[j * 2];
                const float loc_x = my_last[j * 2 + 1];

                const int yc = static_cast<int>(loc_y);
                const int xc = static_cast<int>(loc_x);
                const int y_s = std::max(0, yc - VIEW_RANGE);
                const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
                const int x_s = std::max(0, xc - VIEW_RANGE);
                const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

                for (int ly = y_s; ly < y_e; ++ly) {
                    const int row_off = ly * MAP_SIZE;
                    for (int lx = x_s; lx < x_e; ++lx) {
                        eobs_i[row_off + lx] = 1.0f;
                    }
                }
            }
        }
    }

    static void update_recency(GameStateView& s) {
        for (int i = 0; i < N_AGENTS; ++i) {
            float* agent_recency = s.recency + i * FLAT_MAP_SIZE;

            for (int j = 0; j < FLAT_MAP_SIZE; ++j)
                agent_recency[j] *= RECENCY_DECAY;

            const int yc = static_cast<int>(s.agent_locations[i * 2]);
            const int xc = static_cast<int>(s.agent_locations[i * 2 + 1]);
            const int y_s = std::max(0, yc - VIEW_RANGE);
            const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
            const int x_s = std::max(0, xc - VIEW_RANGE);
            const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

            for (int ly = y_s; ly < y_e; ++ly) {
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

    // Scan only view-range tiles instead of the full map.
    // Running counter for O(1) termination check.
    bool calc_rewards(GameStateView& s,
                      py::detail::unchecked_mutable_reference<float, 2>& rewards,
                      int env_idx) {
        for (int i = 0; i < N_AGENTS; ++i)
            rewards(env_idx, i) = 0.0f;

        int ay[N_AGENTS], ax[N_AGENTS];
        for (int i = 0; i < N_AGENTS; ++i) {
            ay[i] = static_cast<int>(s.agent_locations[i * 2]);
            ax[i] = static_cast<int>(s.agent_locations[i * 2 + 1]);
        }

        for (int i = 0; i < N_AGENTS; ++i) {
            const int y_s = std::max(0, ay[i] - VIEW_RANGE);
            const int y_e = std::min(MAP_SIZE, ay[i] + VIEW_RANGE + 1);
            const int x_s = std::max(0, ax[i] - VIEW_RANGE);
            const int x_e = std::min(MAP_SIZE, ax[i] + VIEW_RANGE + 1);

            for (int y = y_s; y < y_e; ++y) {
                for (int x = x_s; x < x_e; ++x) {
                    const int idx = y * MAP_SIZE + x;
                    if (s.global_discovered[idx] > 0.5f) continue;

                    s.global_discovered[idx] = 1.0f;
                    undiscovered_remaining[env_idx]--;

                    int seeing_count = 0;
                    bool seen_by[N_AGENTS] = {};
                    for (int j = 0; j < N_AGENTS; ++j) {
                        if (std::abs(ay[j] - y) <= VIEW_RANGE &&
                            std::abs(ax[j] - x) <= VIEW_RANGE) {
                            seen_by[j] = true;
                            ++seeing_count;
                        }
                    }

                    const float share = 1.0f / static_cast<float>(seeing_count);
                    for (int j = 0; j < N_AGENTS; ++j) {
                        if (seen_by[j]) rewards(env_idx, j) += share;
                    }
                }
            }
        }

        if (undiscovered_remaining[env_idx] <= 0) {
            for (int i = 0; i < N_AGENTS; ++i) rewards(env_idx, i) += 10.0f;
            return true;
        }
        return false;
    }
};

// --- Pybind11 module ---

PYBIND11_MODULE(_core, m) {
    REGISTER_FEATURE_TYPE_ENUM(m);

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
        .def("get_state", &BatchedEnvironment::get_state)
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
