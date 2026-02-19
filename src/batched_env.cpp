#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <omp.h>
#include <fstream>
#include <string>

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
    GLOBAL_DISCOVERED_FEATURE = 5,
    OTHER_AGENTS_FEATURE = 6,
    OTHER_AGENTS_LAST_KNOWN_FEATURE = 7,
    GLOBAL_UNDISCOVERED_FEATURE = 8,
    OBS_UNDISCOVERED_FEATURE = 9,
    EXPECTED_OBS_UNDISCOVERED_FEATURE = 10
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

// Helper function to compute gravity from other agent positions
// Treats each agent as a point mass with mass=1.0
void get_gravity_from_agents(float* all_agent_locations, int current_agent_idx, int pow, 
                             float agent_x, float agent_y, float& out_gx, float& out_gy) {
    out_gx = 0.0f;
    out_gy = 0.0f;

    for (int j = 0; j < N_AGENTS; ++j) {
        if (j == current_agent_idx) continue; // Skip self
        
        float other_x = all_agent_locations[j * 2 + 1];
        float other_y = all_agent_locations[j * 2];
        
        // Vector from current agent TO other agent
        float dx = other_x - agent_x;
        float dy = other_y - agent_y;
        
        float dist_sq = dx * dx + dy * dy;
        float dist = std::sqrt(dist_sq);
        
        if (dist < 0.1f) continue; // Min distance clamp
        
        // Force magnitude = 1.0 / dist^pow (treating other agent as mass=1.0)
        float denom;
        if (pow == 2) denom = dist * dist_sq;       // dist^3
        else if (pow == 1) denom = dist_sq;         // dist^2
        else denom = std::pow(dist, pow + 1);
        
        float force = 1.0f / denom;
        
        out_gx += dx * force;
        out_gy += dy * force;
    }
}

// Returns vector: Sum(Mass / dist^pow * direction_vector)
// returns dx, dy
// If invert=true, uses (1.0 - map[i]) as mass (for undiscovered features)
void get_gravity(float* map, int pow, float agent_x, float agent_y, float& out_gx, float& out_gy, bool invert = false) {
    out_gx = 0.0f;
    out_gy = 0.0f;

    for (int y = 0; y < MAP_SIZE; ++y) {
        for (int x = 0; x < MAP_SIZE; ++x) {
            float mass = invert ? (1.0f - map[y * MAP_SIZE + x]) : map[y * MAP_SIZE + x];
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
    int seed;
    std::vector<std::mt19937> rngs;
    std::vector<float> data; 
    std::vector<std::string> map_paths;
    std::vector<std::string> expected_map_paths;

    BatchedEnvironment(int n_envs, int sim_seed, 
                       std::vector<std::string> maps = {}, 
                       std::vector<std::string> expected_maps = {}) 
        : num_envs(n_envs), seed(sim_seed), map_paths(maps), expected_map_paths(expected_maps) {
        data.resize(num_envs * ENV_STRIDE);
        rngs.resize(num_envs);
        for(int i=0; i<num_envs; ++i) {
            rngs[i].seed(seed + i);
        }
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
                // Changed to [-1, 1] range
                float val = (std::sin(x * 0.3f + e) + std::cos(y * 0.3f + e*2)) / 2.0f;
                s.actual_danger[i] = std::fmin(1.0f, std::fmax(-1.0f, val));
                 // Midpoint of [-1, 1], will be overwritten by expected logic below
            }
            
            // Expected danger is 0.0 unless we have a prior map.
            if (!expected_map_paths.empty()) {
                std::string path = expected_map_paths[e % expected_map_paths.size()];
                std::ifstream file(path, std::ios::binary);
                if (file.is_open()) {
                    file.read(reinterpret_cast<char*>(s.expected_danger), FLAT_MAP_SIZE * sizeof(float));
                    file.close();
                } else {
                     // Fallback if file load fails
                    for (int i = 0; i < FLAT_MAP_SIZE; ++i) {
                        s.expected_danger[i] = 0.0f; 
                    }
                }
            } else {
                for (int i = 0; i < FLAT_MAP_SIZE; ++i) {
                    s.expected_danger[i] = 0.0f; 
                }
            }

            for (int i = 0; i < N_AGENTS; ++i) {
                s.agent_locations[i * 2] = MAP_SIZE / 2.0f;
                s.agent_locations[i * 2 + 1] = MAP_SIZE / 2.0f;
            }
            
            // Initialize observed_danger with expected_danger for all agents
            for (int i = 0; i < N_AGENTS; ++i) {
                for (int j = 0; j < FLAT_MAP_SIZE; ++j) {
                    s.observed_danger[i * FLAT_MAP_SIZE + j] = s.expected_danger[j];
                }
            }
            
            update_obs(s);
        }
    }

    std::pair<py::array_t<float>, py::array_t<bool>> step(py::array_t<float> actions_array, float communication_prob = -1.0f) {
        auto r = actions_array.unchecked<2>(); 
        py::array_t<float> rewards_array({num_envs, N_AGENTS});
        auto rewards_ptr = rewards_array.mutable_unchecked<2>();
        py::array_t<bool> terminated_array({num_envs});
        auto terminated_ptr = terminated_array.mutable_unchecked<1>();

        // Parallel Step
        #pragma omp parallel for
        for (int e = 0; e < num_envs; ++e) {
            GameStateView s;
            bind_state(s, e);

            update_locations(s, r, e);
            update_obs(s);
            update_last_location(s, e, communication_prob);
            bool done = calc_rewards(s, rewards_ptr, e);
            terminated_ptr(e) = done;
        }

        return {rewards_array, terminated_array};
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
        int pow = 2,
        bool normalize = false) 
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
                bool invert = false;
                
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
                } else if (feature_type == GLOBAL_UNDISCOVERED_FEATURE) {
                    feature_map = s.global_discovered;
                    invert = true; // Use (1.0 - discovered)
                } else if (feature_type == OBS_UNDISCOVERED_FEATURE) {
                    feature_map = s.obs + (i * FLAT_MAP_SIZE);
                    invert = true; // Use (1.0 - obs)
                } else if (feature_type == EXPECTED_OBS_UNDISCOVERED_FEATURE) {
                    feature_map = s.expected_obs + (i * FLAT_MAP_SIZE);
                    invert = true; // Use (1.0 - expected_obs)
                } else if (feature_type == OTHER_AGENTS_FEATURE) {
                    // Use actual current locations of other agents
                    get_gravity_from_agents(s.agent_locations, i, pow, agent_x, agent_y, gx, gy);
                    feature_map = nullptr; // Signal that we already computed gravity
                } else if (feature_type == OTHER_AGENTS_LAST_KNOWN_FEATURE) {
                    // Use this agent's last known locations of other agents
                    float* agent_i_last_known = s.last_agent_locations + (i * 2 * N_AGENTS);
                    get_gravity_from_agents(agent_i_last_known, i, pow, agent_x, agent_y, gx, gy);
                    feature_map = nullptr; // Signal that we already computed gravity
                }

                if (feature_map) {
                    get_gravity(feature_map, pow, agent_x, agent_y, gx, gy, invert);
                } else if (feature_type != OTHER_AGENTS_FEATURE && 
                           feature_type != OTHER_AGENTS_LAST_KNOWN_FEATURE) {
                    gx = gy = 0.0f;
                }

                // Normalize to max norm 1.0 if requested
                if (normalize) {
                    float mag = std::sqrt(gx * gx + gy * gy);
                    if (mag > 1.0f) {
                        gx /= mag;
                        gy /= mag;
                    }
                }

                output_ptr(e, i, 0) = gy; // dy
                output_ptr(e, i, 1) = gx; // dx
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

    void update_last_location(GameStateView& s, int env_idx, float p) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        bool try_radio = (p > 0.0f && p <= 1.0f);

        // For each agent i (viewer/row agent), check if other agents j (target/col agent)
        // are within view range, and update their last known location if so
        for (int i = 0; i < N_AGENTS; ++i) {
            int viewer_y = (int)s.agent_locations[i * 2];
            int viewer_x = (int)s.agent_locations[i * 2 + 1];
            
            for (int j = 0; j < N_AGENTS; ++j) {
                // If it's myself, always update (agents know where they are)
                if (i == j) {
                    s.last_agent_locations[i * (2 * N_AGENTS) + j * 2] = s.agent_locations[j * 2];
                    s.last_agent_locations[i * (2 * N_AGENTS) + j * 2 + 1] = s.agent_locations[j * 2 + 1];
                    continue;
                }

                int target_y = (int)s.agent_locations[j * 2];
                int target_x = (int)s.agent_locations[j * 2 + 1];
                
                bool updated = false;

                // Check physical view range (Visual)
                if (std::abs(viewer_y - target_y) <= VIEW_RANGE && 
                    std::abs(viewer_x - target_x) <= VIEW_RANGE) {
                    updated = true;
                }
                
                // Check radio communication (Probabilistic)
                // "agents should have a probability 'p' of sharing their location to another agent"
                if (!updated && try_radio) {
                    // Agent j shares location with agent i with probability p
                    if (dist(rngs[env_idx]) < p) {
                        updated = true;
                    }
                }

                if (updated) {
                    // Update agent i's knowledge of agent j's location
                    s.last_agent_locations[i * (2 * N_AGENTS) + j * 2] = s.agent_locations[j * 2];
                    s.last_agent_locations[i * (2 * N_AGENTS) + j * 2 + 1] = s.agent_locations[j * 2 + 1];
                }
            }
        }
    }

    // Returns true if all squares are discovered (terminal condition)
    bool calc_rewards(GameStateView& s, py::detail::unchecked_mutable_reference<float, 2>& rewards, int env_idx) {
        for(int i=0; i<N_AGENTS; ++i) rewards(env_idx, i) = 0.0f;

        int undiscovered_count = 0;
        for (int y = 0; y < MAP_SIZE; ++y) {
            for (int x = 0; x < MAP_SIZE; ++x) {
                int idx = y * MAP_SIZE + x;
                if (s.global_discovered[idx] > 0.5f) continue;

                int seeing_count = 0;
                bool seen_by[N_AGENTS] = {false};

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
                } else {
                    undiscovered_count++;
                }
            }
        }

        // All squares discovered: bonus reward and signal termination
        if (undiscovered_count == 0) {
            for (int i = 0; i < N_AGENTS; ++i) rewards(env_idx, i) += 10.0f;
            return true;
        }
        return false;
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
        .value("GLOBAL_DISCOVERED", GLOBAL_DISCOVERED_FEATURE)
        .value("OTHER_AGENTS", OTHER_AGENTS_FEATURE)
        .value("OTHER_AGENTS_LAST_KNOWN", OTHER_AGENTS_LAST_KNOWN_FEATURE)
        .value("GLOBAL_UNDISCOVERED", GLOBAL_UNDISCOVERED_FEATURE)
        .value("OBS_UNDISCOVERED", OBS_UNDISCOVERED_FEATURE)
        .value("EXPECTED_OBS_UNDISCOVERED", EXPECTED_OBS_UNDISCOVERED_FEATURE);

    py::class_<BatchedEnvironment>(m, "BatchedEnvironment")
        .def(py::init<int, int, std::vector<std::string>, std::vector<std::string>>(), 
            py::arg("n_envs"), 
            py::arg("seed") = 42, 
            py::arg("map_paths") = std::vector<std::string>(),
            py::arg("expected_map_paths") = std::vector<std::string>())
        .def("reset", &BatchedEnvironment::reset)
        .def("step", &BatchedEnvironment::step, py::arg("actions"), py::arg("communication_prob") = -1.0f)
        .def("get_memory_view", &BatchedEnvironment::get_memory_view)
        .def("get_stride", &BatchedEnvironment::get_stride)
        .def("get_flat_map_size", &BatchedEnvironment::get_flat_map_size)
        .def("get_gravity_attractions", &BatchedEnvironment::get_gravity_attractions,
             py::arg("agent_mask") = py::none(),
             py::arg("feature_type"),
             py::arg("pow") = 2,
             py::arg("normalize") = false)
        .def_readonly("num_envs", &BatchedEnvironment::num_envs);
}