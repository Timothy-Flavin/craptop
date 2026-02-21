// batched_env_global.cpp - Global-communication mode
// All agents share a single obs mask, single observed_danger map,
// and know each other's true positions.  No expected_obs or
// last_agent_locations needed.
//
// ENV_STRIDE is ~2.4x smaller than partial-obs mode, giving
// significantly better cache locality and fewer per-step computations.
//
// Exposed as pybind11 module: multi_agent_coverage._core_global

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
#ifdef _WIN32
typedef ptrdiff_t ssize_t;
#endif
#endif

namespace py = pybind11;

// --- State layout (global-comms) ---

constexpr int ENV_STRIDE_GLOBAL =
    FLAT_MAP_SIZE +              // expected_danger   (prior belief)
    FLAT_MAP_SIZE +              // actual_danger     (ground truth)
    FLAT_MAP_SIZE +              // observed_danger   (shared across agents)
    FLAT_MAP_SIZE +              // obs               (shared = global_discovered)
    (N_AGENTS * 2) +             // agent_locations   (true positions, known to all)
    (N_AGENTS * FLAT_MAP_SIZE) + // recency           (per-agent)
    N_AGENTS;                    // agents_alive      (ground truth, 1.0=alive 0.0=dead)

struct GameStateViewGlobal
{
    float *expected_danger;
    float *actual_danger;
    float *observed_danger;
    float *obs; // shared observation mask (= global_discovered)
    float *agent_locations;
    float *recency;
    float *agents_alive;
};

// --- BatchedEnvironmentGlobal ---

class BatchedEnvironmentGlobal
{
public:
    int num_envs;
    int seed;
    std::vector<std::mt19937> rngs;
    std::vector<float> data;
    std::vector<std::string> map_paths;
    std::vector<std::string> expected_map_paths;
    std::vector<bool> env_terminated;
    std::vector<int> undiscovered_remaining;
    bool reset_automatically;
    float death_penalty;

    BatchedEnvironmentGlobal(int n_envs, int sim_seed,
                             std::vector<std::string> maps = {},
                             std::vector<std::string> expected_maps = {},
                             bool auto_reset = true,
                             float death_penalty_val = -20.0f)
        : num_envs(n_envs), seed(sim_seed), map_paths(std::move(maps)),
          expected_map_paths(std::move(expected_maps)), reset_automatically(auto_reset),
          death_penalty(death_penalty_val)
    {
        data.resize(static_cast<size_t>(num_envs) * ENV_STRIDE_GLOBAL, 0.0f);
        rngs.resize(num_envs);
        env_terminated.assign(num_envs, false);
        undiscovered_remaining.assign(num_envs, FLAT_MAP_SIZE);
        for (int i = 0; i < num_envs; ++i)
            rngs[i].seed(seed + i);
        reset();
    }

    void bind_state(GameStateViewGlobal &s, int env_idx)
    {
        float *ptr = data.data() + (static_cast<size_t>(env_idx) * ENV_STRIDE_GLOBAL);
        s.expected_danger = ptr;
        ptr += FLAT_MAP_SIZE;
        s.actual_danger = ptr;
        ptr += FLAT_MAP_SIZE;
        s.observed_danger = ptr;
        ptr += FLAT_MAP_SIZE;
        s.obs = ptr;
        ptr += FLAT_MAP_SIZE;
        s.agent_locations = ptr;
        ptr += N_AGENTS * 2;
        s.recency = ptr;
        ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.agents_alive = ptr;
    }

    void reset_env(GameStateViewGlobal &s, int e)
    {
        std::memset(data.data() + (static_cast<size_t>(e) * ENV_STRIDE_GLOBAL), 0,
                    ENV_STRIDE_GLOBAL * sizeof(float));

        // Actual danger
        if (!map_paths.empty())
        {
            const std::string &path = map_paths[e % map_paths.size()];
            std::ifstream file(path, std::ios::binary);
            if (file.is_open())
            {
                file.read(reinterpret_cast<char *>(s.actual_danger), FLAT_MAP_SIZE * sizeof(float));
            }
            else
            {
                generate_procedural_danger(s.actual_danger, e);
            }
        }
        else
        {
            generate_procedural_danger(s.actual_danger, e);
        }

        // Expected danger (prior belief / satellite data)
        if (!expected_map_paths.empty())
        {
            const std::string &path = expected_map_paths[e % expected_map_paths.size()];
            std::ifstream file(path, std::ios::binary);
            if (file.is_open())
            {
                file.read(reinterpret_cast<char *>(s.expected_danger), FLAT_MAP_SIZE * sizeof(float));
            }
        }

        // Agent locations: center of map
        const float center = MAP_SIZE * 0.5f;
        for (int i = 0; i < N_AGENTS; ++i)
        {
            s.agent_locations[i * 2] = center;
            s.agent_locations[i * 2 + 1] = center;
        }

        // Init observed_danger from expected_danger (shared, single copy)
        std::memcpy(s.observed_danger, s.expected_danger, FLAT_MAP_SIZE * sizeof(float));

        // Init agents_alive = 1.0
        for (int i = 0; i < N_AGENTS; ++i)
            s.agents_alive[i] = 1.0f;

        // Discover initial view and set obs
        undiscovered_remaining[e] = FLAT_MAP_SIZE;
        discover_and_update_obs(s, e);
    }

    void reset()
    {
        env_terminated.assign(num_envs, false);
#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            GameStateViewGlobal s;
            bind_state(s, e);
            reset_env(s, e);
        }
    }

    // Reset a single environment by index.  Thread-safe as long as no
    // concurrent step() call is operating on the same env_idx.
    void reset_single(int env_idx)
    {
        if (env_idx < 0 || env_idx >= num_envs)
            throw std::out_of_range("reset_single: env_idx out of range");
        GameStateViewGlobal s;
        bind_state(s, env_idx);
        reset_env(s, env_idx);
        env_terminated[env_idx] = false;
    }

    std::pair<py::array_t<float>, py::array_t<bool>> step(
        py::array_t<float> actions_array)
    {
        auto r = actions_array.unchecked<2>();
        py::array_t<float> rewards_array({num_envs, N_AGENTS});
        auto rewards_ptr = rewards_array.mutable_unchecked<2>();
        py::array_t<bool> terminated_array({num_envs});
        auto terminated_ptr = terminated_array.mutable_unchecked<1>();

#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            GameStateViewGlobal s;
            bind_state(s, e);

            if (env_terminated[e])
            {
                if (reset_automatically)
                {
                    reset_env(s, e);
                    env_terminated[e] = false;
                }
                else
                {
                    // Hold: return zero rewards and keep state frozen
                    for (int i = 0; i < N_AGENTS; ++i)
                        rewards_ptr(e, i) = 0.0f;
                    terminated_ptr(e) = true;
                    continue;
                }
            }

            update_locations(s, r, e);

            // Death roll: alive agents on danger > 0 tiles may die
            for (int i = 0; i < N_AGENTS; ++i)
            {
                if (s.agents_alive[i] < 0.5f)
                    continue;
                const int cy = std::clamp(static_cast<int>(s.agent_locations[i * 2]), 0, MAP_SIZE - 1);
                const int cx = std::clamp(static_cast<int>(s.agent_locations[i * 2 + 1]), 0, MAP_SIZE - 1);
                const float danger = s.actual_danger[cy * MAP_SIZE + cx];
                if (danger > 0.0f)
                {
                    const float p_death = danger / DANGER_FACTOR;
                    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
                    if (dist01(rngs[e]) < p_death)
                    {
                        s.agents_alive[i] = 0.0f;
                        rewards_ptr(e, i) += death_penalty;
                    }
                }
            }

            // Combined discover + obs update + rewards
            bool done = discover_reward_and_update_obs(s, rewards_ptr, e);
            update_recency(s);
            terminated_ptr(e) = done;
            env_terminated[e] = done;
        }
        return {rewards_array, terminated_array};
    }

    std::pair<size_t, size_t> get_memory_view()
    {
        return {reinterpret_cast<size_t>(data.data()), data.size() * sizeof(float)};
    }

    // Returns a writable numpy array that is a zero-copy view of the state buffer.
    // Python can read and modify state in-place; no data is copied.
    py::array_t<float> get_state()
    {
        py::capsule base(data.data(), [](void *) {}); // no-op: memory owned by C++
        return py::array_t<float>(
            {static_cast<ssize_t>(num_envs), static_cast<ssize_t>(ENV_STRIDE_GLOBAL)},
            {static_cast<ssize_t>(ENV_STRIDE_GLOBAL) * static_cast<ssize_t>(sizeof(float)),
             static_cast<ssize_t>(sizeof(float))},
            data.data(),
            base);
    }

    int get_stride() const { return ENV_STRIDE_GLOBAL; }
    int get_flat_map_size() const { return FLAT_MAP_SIZE; }

    py::array_t<float> get_gravity_attractions(
        py::object agent_mask_obj,
        int feature_type,
        int pow = 2,
        bool normalize = false,
        bool local = false)
    {
        bool agent_mask[N_AGENTS];
        if (agent_mask_obj.is_none())
        {
            for (int i = 0; i < N_AGENTS; ++i)
                agent_mask[i] = true;
        }
        else
        {
            auto mask = agent_mask_obj.cast<py::array_t<bool>>();
            auto mask_ptr = mask.data();
            for (int i = 0; i < N_AGENTS; ++i)
                agent_mask[i] = mask_ptr[i];
        }

        py::array_t<float> output_array({num_envs, N_AGENTS, 2});
        auto output_ptr = output_array.mutable_unchecked<3>();

#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            GameStateViewGlobal s;
            bind_state(s, e);

            for (int i = 0; i < N_AGENTS; ++i)
            {
                if (!agent_mask[i])
                {
                    output_ptr(e, i, 0) = 0.0f;
                    output_ptr(e, i, 1) = 0.0f;
                    continue;
                }

                const float agent_y = s.agent_locations[i * 2];
                const float agent_x = s.agent_locations[i * 2 + 1];
                float gx = 0.0f, gy = 0.0f;

                switch (static_cast<FeatureType>(feature_type))
                {
                case EXPECTED_DANGER_FEATURE:
                    dispatch_map_gravity(s.expected_danger, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case ACTUAL_DANGER_FEATURE:
                    dispatch_map_gravity(s.actual_danger, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case OBSERVED_DANGER_FEATURE:
                    dispatch_map_gravity(s.observed_danger, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                // OBS = EXPECTED_OBS = GLOBAL_DISCOVERED in global mode
                case OBS_FEATURE:
                case EXPECTED_OBS_FEATURE:
                case GLOBAL_DISCOVERED_FEATURE:
                    dispatch_map_gravity(s.obs, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case RECENCY_FEATURE:
                    dispatch_map_gravity(s.recency + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, false, local);
                    break;
                case GLOBAL_UNDISCOVERED_FEATURE:
                case OBS_UNDISCOVERED_FEATURE:
                case EXPECTED_OBS_UNDISCOVERED_FEATURE:
                    dispatch_map_gravity(s.obs, pow, agent_x, agent_y, gx, gy, true, local);
                    break;
                case RECENCY_STALE_FEATURE:
                    dispatch_map_gravity(s.recency + i * FLAT_MAP_SIZE, pow, agent_x, agent_y, gx, gy, true, local);
                    break;
                // OTHER_AGENTS = OTHER_AGENTS_LAST_KNOWN (all positions known)
                case OTHER_AGENTS_FEATURE:
                case OTHER_AGENTS_LAST_KNOWN_FEATURE:
                    if (local)
                        get_gravity_from_agents_local(s.agent_locations, i, pow, agent_x, agent_y, gx, gy);
                    else
                        get_gravity_from_agents(s.agent_locations, i, pow, agent_x, agent_y, gx, gy);
                    break;
                case WALL_REPEL_FEATURE:
                    get_gravity_wall(pow, agent_x, agent_y, gx, gy, local);
                    gx = -gx;
                    gy = -gy;
                    break;
                case WALL_ATTRACT_FEATURE:
                    get_gravity_wall(pow, agent_x, agent_y, gx, gy, local);
                    break;
                // In global mode both Voronoi variants use the shared obs + true positions
                case GLOBAL_VORONOI_UNDISCOVERED_FEATURE:
                case EXPECTED_VORONOI_UNDISCOVERED_FEATURE:
                    if (local)
                        get_gravity_voronoi_local(s.obs, s.agent_locations, i, pow, agent_x, agent_y, gx, gy, s.agents_alive);
                    else
                        get_gravity_voronoi(s.obs, s.agent_locations, i, pow, agent_x, agent_y, gx, gy, s.agents_alive);
                    break;
                default:
                    gx = gy = 0.0f;
                    break;
                }

                if (normalize)
                {
                    const float mag = std::sqrt(gx * gx + gy * gy);
                    if (mag > 1.0f)
                    {
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
    void update_locations(GameStateViewGlobal &s,
                          const py::detail::unchecked_reference<float, 2> &actions,
                          int env_idx)
    {
        for (int i = 0; i < N_AGENTS; ++i)
        {
            // Dead agents don't move
            if (s.agents_alive[i] < 0.5f)
                continue;

            float dy = actions(env_idx, i * 2);
            float dx = actions(env_idx, i * 2 + 1);

            const float len_sq = dy * dy + dx * dx;
            if (len_sq > 0.00000001f)
            {
                const float inv_len = 1.0f / std::sqrt(len_sq);
                dy *= inv_len;
                dx *= inv_len;
            }

            const int cy = std::clamp(static_cast<int>(s.agent_locations[i * 2]), 0, MAP_SIZE - 1);
            const int cx = std::clamp(static_cast<int>(s.agent_locations[i * 2 + 1]), 0, MAP_SIZE - 1);
            const float danger = s.actual_danger[cy * MAP_SIZE + cx];
            const float effective_speed = SPEED * (1.0f - danger * DANGER_PENALTY_FACTOR);

            s.agent_locations[i * 2] = std::clamp(s.agent_locations[i * 2] + dy * effective_speed, 0.0f, MAP_MAX);
            s.agent_locations[i * 2 + 1] = std::clamp(s.agent_locations[i * 2 + 1] + dx * effective_speed, 0.0f, MAP_MAX);
        }
    }

    // Used on reset: discover initial tiles, set obs + observed_danger, no rewards.
    void discover_and_update_obs(GameStateViewGlobal &s, int e)
    {
        for (int i = 0; i < N_AGENTS; ++i)
        {
            const int yc = static_cast<int>(s.agent_locations[i * 2]);
            const int xc = static_cast<int>(s.agent_locations[i * 2 + 1]);
            const int y_s = std::max(0, yc - VIEW_RANGE);
            const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
            const int x_s = std::max(0, xc - VIEW_RANGE);
            const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

            for (int ly = y_s; ly < y_e; ++ly)
            {
                const int row_off = ly * MAP_SIZE;
                for (int lx = x_s; lx < x_e; ++lx)
                {
                    const int idx = row_off + lx;
                    if (s.obs[idx] < 0.5f)
                    {
                        s.obs[idx] = 1.0f;
                        undiscovered_remaining[e]--;
                    }
                    s.observed_danger[idx] = s.actual_danger[idx];
                }
            }
        }
    }

    // Combined: discover new tiles, compute rewards, update obs + observed_danger.
    bool discover_reward_and_update_obs(
        GameStateViewGlobal &s,
        py::detail::unchecked_mutable_reference<float, 2> &rewards,
        int env_idx)
    {
        for (int i = 0; i < N_AGENTS; ++i)
            rewards(env_idx, i) = 0.0f;

        int ay[N_AGENTS], ax[N_AGENTS];
        for (int i = 0; i < N_AGENTS; ++i)
        {
            ay[i] = static_cast<int>(s.agent_locations[i * 2]);
            ax[i] = static_cast<int>(s.agent_locations[i * 2 + 1]);
        }

        for (int i = 0; i < N_AGENTS; ++i)
        {
            const int y_s = std::max(0, ay[i] - VIEW_RANGE);
            const int y_e = std::min(MAP_SIZE, ay[i] + VIEW_RANGE + 1);
            const int x_s = std::max(0, ax[i] - VIEW_RANGE);
            const int x_e = std::min(MAP_SIZE, ax[i] + VIEW_RANGE + 1);

            for (int y = y_s; y < y_e; ++y)
            {
                const int row_off = y * MAP_SIZE;
                for (int x = x_s; x < x_e; ++x)
                {
                    const int idx = row_off + x;
                    // Always update observed_danger for visible tiles
                    s.observed_danger[idx] = s.actual_danger[idx];

                    if (s.obs[idx] > 0.5f)
                        continue; // already discovered

                    // Newly discovered tile
                    s.obs[idx] = 1.0f;
                    undiscovered_remaining[env_idx]--;

                    // Count all agents seeing this tile
                    int seeing_count = 0;
                    bool seen_by[N_AGENTS] = {};
                    for (int j = 0; j < N_AGENTS; ++j)
                    {
                        if (std::abs(ay[j] - y) <= VIEW_RANGE &&
                            std::abs(ax[j] - x) <= VIEW_RANGE)
                        {
                            seen_by[j] = true;
                            ++seeing_count;
                        }
                    }

                    const float share = 1.0f / static_cast<float>(seeing_count);
                    for (int j = 0; j < N_AGENTS; ++j)
                    {
                        if (seen_by[j])
                            rewards(env_idx, j) += share;
                    }
                }
            }
        }

        if (undiscovered_remaining[env_idx] <= 0)
        {
            for (int i = 0; i < N_AGENTS; ++i)
                rewards(env_idx, i) += 10.0f;
            return true;
        }
        return false;
    }

    static void update_recency(GameStateViewGlobal &s)
    {
        for (int i = 0; i < N_AGENTS; ++i)
        {
            float *agent_recency = s.recency + i * FLAT_MAP_SIZE;

            for (int j = 0; j < FLAT_MAP_SIZE; ++j)
                agent_recency[j] *= RECENCY_DECAY;

            const int yc = static_cast<int>(s.agent_locations[i * 2]);
            const int xc = static_cast<int>(s.agent_locations[i * 2 + 1]);
            const int y_s = std::max(0, yc - VIEW_RANGE);
            const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
            const int x_s = std::max(0, xc - VIEW_RANGE);
            const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

            for (int ly = y_s; ly < y_e; ++ly)
            {
                float *row_start = agent_recency + ly * MAP_SIZE + x_s;
                const int run_len = x_e - x_s;
                for (int k = 0; k < run_len; ++k)
                    row_start[k] = 1.0f;
            }
        }
    }
};

// --- Pybind11 module ---

PYBIND11_MODULE(_core_global, m)
{
    // FeatureType enum is already registered by _core – do NOT re-register here.

    py::class_<BatchedEnvironmentGlobal>(m, "BatchedEnvironment")
        .def(py::init<int, int, std::vector<std::string>, std::vector<std::string>, bool, float>(),
             py::arg("n_envs"),
             py::arg("seed") = 42,
             py::arg("map_paths") = std::vector<std::string>(),
             py::arg("expected_map_paths") = std::vector<std::string>(),
             py::arg("reset_automatically") = true,
             py::arg("death_penalty") = -20.0f)
        .def("reset", &BatchedEnvironmentGlobal::reset)
        .def("reset_single", &BatchedEnvironmentGlobal::reset_single, py::arg("env_idx"))
        .def_readwrite("reset_automatically", &BatchedEnvironmentGlobal::reset_automatically)
        .def_readwrite("death_penalty", &BatchedEnvironmentGlobal::death_penalty)
        .def("step", &BatchedEnvironmentGlobal::step,
             py::arg("actions"))
        .def("get_memory_view", &BatchedEnvironmentGlobal::get_memory_view)
        .def("get_state", &BatchedEnvironmentGlobal::get_state)
        .def("get_stride", &BatchedEnvironmentGlobal::get_stride)
        .def("get_flat_map_size", &BatchedEnvironmentGlobal::get_flat_map_size)
        .def("get_gravity_attractions", &BatchedEnvironmentGlobal::get_gravity_attractions,
             py::arg("agent_mask") = py::none(),
             py::arg("feature_type"),
             py::arg("pow") = 2,
             py::arg("normalize") = false,
             py::arg("local") = false)
        .def_readonly("num_envs", &BatchedEnvironmentGlobal::num_envs);
}
