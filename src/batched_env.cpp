#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <omp.h>

namespace py = pybind11;

#define MAP_SIZE 32
#define N_AGENTS 4
#define VIEW_RANGE 3
#define SPEED 0.5f
#define DANGER_PENALTY_FACTOR 0.8f

// Feature type enum
enum FeatureType {
    EXPECTED_DANGER_FEATURE = 0,
    ACTUAL_DANGER_FEATURE = 1,
    OBSERVED_DANGER_FEATURE = 2,
    OBS_FEATURE = 3,
    EXPECTED_OBS_FEATURE = 4,
    GLOBAL_DISCOVERED_FEATURE = 5
};

constexpr int FLAT_MAP_SIZE = (MAP_SIZE * MAP_SIZE);

// Total floats per environment
constexpr int ENV_STRIDE = 
    FLAT_MAP_SIZE +                 // expected_danger (0)
    FLAT_MAP_SIZE +                 // actual_danger (1024)
    (N_AGENTS * FLAT_MAP_SIZE) +    // observed_danger (2048)
    (N_AGENTS * FLAT_MAP_SIZE) +    // obs (mask) (6144)
    (N_AGENTS * 2) +                // agent_locations (10240)
    (N_AGENTS * FLAT_MAP_SIZE) +    // expected_obs (10248)
    (N_AGENTS * 2 * N_AGENTS) +     // last_agent_locations (14344)
    FLAT_MAP_SIZE;                  // global_discovered (14376)
    // Total size approx 15400 floats per env

struct GameStateView {
    float* expected_danger;   
    float* actual_danger;     
    float* observed_danger;   
    float* obs;               
    float* agent_locations;   
    float* expected_obs;      
    float* last_agent_locations; 
    float* global_discovered; 
};

// Returns vector: Sum(Mass / dist^pow * direction_vector)
// returns dx, dy
void get_gravity(float* map, int pow, float agent_x, float agent_y, float& out_gx, float& out_gy) {
    out_gx = 0.0f;
    out_gy = 0.0f;

    for (int y = 0; y < MAP_SIZE; ++y) {
        for (int x = 0; x < MAP_SIZE; ++x) {
            float mass = map[y * MAP_SIZE + x];
            if (mass <= 0.001f) continue; // Optimization: Ignore empty cells

            // Vector from Agent TO Cell
            float dx = (float)x - agent_x;
            float dy = (float)y - agent_y;

            float dist_sq = dx * dx + dy * dy;
            float dist = std::sqrt(dist_sq);

            if (dist < 0.1f) continue; // Min distance clamp

            // Force magnitude = Mass / dist^pow
            // Since we need to multiply by normalized direction (dx/dist),
            // We actually divide by dist^(pow+1)
            
            float denom;
            if (pow == 2) denom = dist * dist_sq;       // dist^3
            else if (pow == 1) denom = dist_sq;         // dist^2
            else denom = std::pow(dist, pow + 1);
            
            float force = mass / denom;

            out_gx += dx * force;
            out_gy += dy * force;
        }
    }
}

class BatchedEnvironment {
public:
    int num_envs;
    std::vector<float> data; 

    BatchedEnvironment(int n_envs) : num_envs(n_envs) {
        data.resize(num_envs * ENV_STRIDE);
        reset();
    }

    void bind_state(GameStateView& s, int env_idx) {
        float* ptr = data.data() + (env_idx * ENV_STRIDE);
        s.expected_danger = ptr; ptr += FLAT_MAP_SIZE;
        s.actual_danger = ptr; ptr += FLAT_MAP_SIZE;
        s.observed_danger = ptr; ptr += (N_AGENTS * FLAT_MAP_SIZE);
        s.obs = ptr; ptr += (N_AGENTS * FLAT_MAP_SIZE);
        s.agent_locations = ptr; ptr += (N_AGENTS * 2);
        s.expected_obs = ptr; ptr += (N_AGENTS * FLAT_MAP_SIZE);
        s.last_agent_locations = ptr; ptr += (N_AGENTS * 2 * N_AGENTS);
        s.global_discovered = ptr; 
    }

    void reset() {
        // Parallel reset
        #pragma omp parallel for
        for (int e = 0; e < num_envs; ++e) {
            GameStateView s;
            bind_state(s, e);
            
            // Zero out memory for this env
            std::memset(data.data() + (e * ENV_STRIDE), 0, ENV_STRIDE * sizeof(float));

            // Procedural Map Gen (Thread-safe RNG is tricky, using simple math here)
            for (int i = 0; i < FLAT_MAP_SIZE; ++i) {
                int y = i / MAP_SIZE;
                int x = i % MAP_SIZE;
                // Deterministic pseudo-random based on env index
                float val = (std::sin(x * 0.3f + e) + std::cos(y * 0.3f + e*2) + 2.0f) / 4.0f;
                s.actual_danger[i] = std::fmin(1.0f, std::fmax(0.0f, val));
                s.expected_danger[i] = 0.5f; 
            }

            for (int i = 0; i < N_AGENTS; ++i) {
                s.agent_locations[i * 2] = MAP_SIZE / 2.0f;
                s.agent_locations[i * 2 + 1] = MAP_SIZE / 2.0f;
            }
            
            update_obs(s);
        }
    }

    py::array_t<float> step(py::array_t<float> actions_array) {
        auto r = actions_array.unchecked<2>(); 
        py::array_t<float> rewards_array({num_envs, N_AGENTS});
        auto rewards_ptr = rewards_array.mutable_unchecked<2>();

        // Parallel Step
        #pragma omp parallel for
        for (int e = 0; e < num_envs; ++e) {
            GameStateView s;
            bind_state(s, e);

            update_locations(s, r, e);
            update_obs(s);
            calc_rewards(s, rewards_ptr, e);
        }

        return rewards_array;
    }

    // Return (memory_ptr, size_bytes) for Python ctypes/torch
    std::pair<size_t, size_t> get_memory_view() {
        return { (size_t)data.data(), data.size() * sizeof(float) };
    }

    int get_stride() { return ENV_STRIDE; }
    int get_flat_map_size() { return FLAT_MAP_SIZE; }

    py::array_t<float> get_gravity_attractions(
        py::object agent_mask_obj, 
        int feature_type,
        int pow = 2) 
    {
        // Parse agent mask (None means all agents)
        std::vector<bool> agent_mask(N_AGENTS, true);
        if (!agent_mask_obj.is_none()) {
            auto mask = agent_mask_obj.cast<py::array_t<bool>>();
            auto mask_ptr = mask.data();
            for (int i = 0; i < N_AGENTS; ++i) {
                agent_mask[i] = mask_ptr[i];
            }
        }

        // Allocate output: (num_envs, N_AGENTS, 2)
        py::array_t<float> output_array({num_envs, N_AGENTS, 2});
        auto output_ptr = output_array.mutable_unchecked<3>();

        #pragma omp parallel for
        for (int e = 0; e < num_envs; ++e) {
            GameStateView s;
            bind_state(s, e);

            for (int i = 0; i < N_AGENTS; ++i) {
                if (!agent_mask[i]) {
                    output_ptr(e, i, 0) = 0.0f;
                    output_ptr(e, i, 1) = 0.0f;
                    continue;
                }

                float agent_x = s.agent_locations[i * 2 + 1];
                float agent_y = s.agent_locations[i * 2];
                float gx, gy;

                float* feature_map = nullptr;
                
                if (feature_type == EXPECTED_DANGER_FEATURE) {
                    feature_map = s.expected_danger;
                } else if (feature_type == ACTUAL_DANGER_FEATURE) {
                    feature_map = s.actual_danger;
                } else if (feature_type == OBSERVED_DANGER_FEATURE) {
                    feature_map = s.observed_danger + (i * FLAT_MAP_SIZE);
                } else if (feature_type == OBS_FEATURE) {
                    feature_map = s.obs + (i * FLAT_MAP_SIZE);
                } else if (feature_type == EXPECTED_OBS_FEATURE) {
                    feature_map = s.expected_obs + (i * FLAT_MAP_SIZE);
                } else if (feature_type == GLOBAL_DISCOVERED_FEATURE) {
                    feature_map = s.global_discovered;
                }

                if (feature_map) {
                    get_gravity(feature_map, pow, agent_x, agent_y, gx, gy);
                } else {
                    gx = gy = 0.0f;
                }

                output_ptr(e, i, 0) = gx;
                output_ptr(e, i, 1) = gy;
            }
        }

        return output_array;
    }

private:
    void update_locations(GameStateView& s, const py::detail::unchecked_reference<float, 2>& actions, int env_idx) {
        for (int i = 0; i < N_AGENTS; ++i) {
            float dy = actions(env_idx, i * 2);
            float dx = actions(env_idx, i * 2 + 1);
            float len = std::sqrt(dy * dy + dx * dx);
            if (len > 0.0001f) { dy /= len; dx /= len; }

            int cy = (int)s.agent_locations[i * 2];
            int cx = (int)s.agent_locations[i * 2 + 1];
            
            cy = std::max(0, std::min(MAP_SIZE - 1, cy));
            cx = std::max(0, std::min(MAP_SIZE - 1, cx));
            
            float danger = s.actual_danger[cy * MAP_SIZE + cx];
            float effective_speed = SPEED * (1.0f - (danger * DANGER_PENALTY_FACTOR));

            float ny = s.agent_locations[i * 2] + dy * effective_speed;
            float nx = s.agent_locations[i * 2 + 1] + dx * effective_speed;

            s.agent_locations[i * 2] = std::fmax(0.0f, std::fmin((float)MAP_SIZE - 0.01f, ny));
            s.agent_locations[i * 2 + 1] = std::fmax(0.0f, std::fmin((float)MAP_SIZE - 0.01f, nx));
        }
    }

    void update_obs(GameStateView& s) {
        for (int i = 0; i < N_AGENTS; ++i) {
            int yc = (int)s.agent_locations[i * 2];
            int xc = (int)s.agent_locations[i * 2 + 1];

            int y_s = std::max(0, yc - VIEW_RANGE);
            int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
            int x_s = std::max(0, xc - VIEW_RANGE);
            int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

            for (int ly = y_s; ly < y_e; ++ly) {
                for (int lx = x_s; lx < x_e; ++lx) {
                    int idx = ly * MAP_SIZE + lx;
                    int agent_idx = i * FLAT_MAP_SIZE + idx;
                    s.obs[agent_idx] = 1.0f;
                    s.observed_danger[agent_idx] = s.actual_danger[idx];
                }
            }
        }
    }

    void calc_rewards(GameStateView& s, py::detail::unchecked_mutable_reference<float, 2>& rewards, int env_idx) {
        for(int i=0; i<N_AGENTS; ++i) rewards(env_idx, i) = 0.0f;

        for (int y = 0; y < MAP_SIZE; ++y) {
            for (int x = 0; x < MAP_SIZE; ++x) {
                int idx = y * MAP_SIZE + x;
                if (s.global_discovered[idx] > 0.5f) continue;

                int seeing_count = 0;
                bool seen_by[N_AGENTS] = {false}; // Fixed size array initialization

                for (int i = 0; i < N_AGENTS; ++i) {
                    int ay = (int)s.agent_locations[i * 2];
                    int ax = (int)s.agent_locations[i * 2 + 1];
                    if (std::abs(ay - y) <= VIEW_RANGE && std::abs(ax - x) <= VIEW_RANGE) {
                        seen_by[i] = true;
                        seeing_count++;
                    }
                }

                if (seeing_count > 0) {
                    s.global_discovered[idx] = 1.0f;
                    float share = 1.0f / (float)seeing_count;
                    for (int i = 0; i < N_AGENTS; ++i) {
                        if (seen_by[i]) rewards(env_idx, i) += share;
                    }
                }
            }
        }
    }
};

PYBIND11_MODULE(multi_agent_coverage, m) {
    // Feature type enum
    py::enum_<FeatureType>(m, "FeatureType")
        .value("EXPECTED_DANGER", EXPECTED_DANGER_FEATURE)
        .value("ACTUAL_DANGER", ACTUAL_DANGER_FEATURE)
        .value("OBSERVED_DANGER", OBSERVED_DANGER_FEATURE)
        .value("OBS", OBS_FEATURE)
        .value("EXPECTED_OBS", EXPECTED_OBS_FEATURE)
        .value("GLOBAL_DISCOVERED", GLOBAL_DISCOVERED_FEATURE);

    py::class_<BatchedEnvironment>(m, "BatchedEnvironment")
        .def(py::init<int>())
        .def("reset", &BatchedEnvironment::reset)
        .def("step", &BatchedEnvironment::step)
        .def("get_memory_view", &BatchedEnvironment::get_memory_view)
        .def("get_stride", &BatchedEnvironment::get_stride)
        .def("get_flat_map_size", &BatchedEnvironment::get_flat_map_size)
        .def("get_gravity_attractions", &BatchedEnvironment::get_gravity_attractions,
             py::arg("agent_mask") = py::none(),
             py::arg("feature_type"),
             py::arg("pow") = 2)
        .def_readonly("num_envs", &BatchedEnvironment::num_envs);
}