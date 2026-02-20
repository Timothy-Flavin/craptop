#pragma once
// gravity.h – shared constants, enums, and force-calculation helpers
// Included by both batched_env.cpp (partial-obs) and batched_env_global.cpp (global-comms).

#include <cmath>
#include <algorithm>

// ─── Shared constants ───────────────────────────────────────────────────────

constexpr int MAP_SIZE = 32;
constexpr int N_AGENTS = 4;
constexpr int VIEW_RANGE = 3;
constexpr float SPEED = 0.5f;
constexpr float DANGER_PENALTY_FACTOR = 0.8f;
constexpr int FLAT_MAP_SIZE = MAP_SIZE * MAP_SIZE;
constexpr float RECENCY_DECAY = 0.99f;
constexpr float MAP_MAX = MAP_SIZE - 0.01f;
constexpr float DIST_SQ_MIN = 0.001f;

// ─── Feature type enum (identical in both modes) ────────────────────────────

enum FeatureType
{
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
    WALL_ATTRACT_FEATURE = 14,
    GLOBAL_VORONOI_UNDISCOVERED_FEATURE = 15,
    EXPECTED_VORONOI_UNDISCOVERED_FEATURE = 16
};

// ─── Gravity helpers ────────────────────────────────────────────────────────
// All marked inline so the header can be included by multiple TUs.

// Gravity from a map (global). Iterates all MAP_SIZE*MAP_SIZE tiles.
inline void get_gravity(const float *__restrict map, int pow,
                        float agent_x, float agent_y,
                        float &out_gx, float &out_gy, bool invert)
{
    float gx = 0.0f, gy = 0.0f;

    for (int y = 0; y < MAP_SIZE; ++y)
    {
        const float dy = static_cast<float>(y) - agent_y;
        const float *row = map + y * MAP_SIZE;
        for (int x = 0; x < MAP_SIZE; ++x)
        {
            float mass = invert ? (1.0f - row[x]) : row[x];
            if (mass <= 0.001f)
                continue;

            const float dx = static_cast<float>(x) - agent_x;
            const float dist_sq = dx * dx + dy * dy;
            if (dist_sq < DIST_SQ_MIN)
                continue;

            const float dist = std::sqrt(dist_sq);
            float denom;
            switch (pow)
            {
            case 1:
                denom = dist_sq;
                break;
            case 2:
                denom = dist * dist_sq;
                break;
            default:
            {
                denom = dist;
                for (int p = 0; p < pow; ++p)
                    denom *= dist;
                break;
            }
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
inline void get_gravity_local(const float *__restrict map, int pow,
                              float agent_x, float agent_y,
                              float &out_gx, float &out_gy, bool invert)
{
    float gx = 0.0f, gy = 0.0f;

    const int yc = static_cast<int>(agent_y);
    const int xc = static_cast<int>(agent_x);
    const int y_s = std::max(0, yc - VIEW_RANGE);
    const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
    const int x_s = std::max(0, xc - VIEW_RANGE);
    const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

    for (int y = y_s; y < y_e; ++y)
    {
        const float dy = static_cast<float>(y) - agent_y;
        const float *row = map + y * MAP_SIZE;
        for (int x = x_s; x < x_e; ++x)
        {
            float mass = invert ? (1.0f - row[x]) : row[x];
            if (mass <= 0.001f)
                continue;

            const float dx = static_cast<float>(x) - agent_x;
            const float dist_sq = dx * dx + dy * dy;
            if (dist_sq < DIST_SQ_MIN)
                continue;

            const float dist = std::sqrt(dist_sq);
            float denom;
            switch (pow)
            {
            case 1:
                denom = dist_sq;
                break;
            case 2:
                denom = dist * dist_sq;
                break;
            default:
            {
                denom = dist;
                for (int p = 0; p < pow; ++p)
                    denom *= dist;
                break;
            }
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
inline void get_gravity_from_agents(const float *__restrict all_locs, int skip_idx, int pow,
                                    float agent_x, float agent_y,
                                    float &out_gx, float &out_gy)
{
    float gx = 0.0f, gy = 0.0f;

    for (int j = 0; j < N_AGENTS; ++j)
    {
        if (j == skip_idx)
            continue;
        const float other_y = all_locs[j * 2];
        const float other_x = all_locs[j * 2 + 1];
        const float dx = other_x - agent_x;
        const float dy = other_y - agent_y;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq < DIST_SQ_MIN)
            continue;
        const float dist = std::sqrt(dist_sq);
        float denom;
        switch (pow)
        {
        case 1:
            denom = dist_sq;
            break;
        case 2:
            denom = dist * dist_sq;
            break;
        default:
        {
            denom = dist;
            for (int p = 0; p < pow; ++p)
                denom *= dist;
            break;
        }
        }
        const float f = 1.0f / denom;
        gx += dx * f;
        gy += dy * f;
    }
    out_gx = gx;
    out_gy = gy;
}

// Gravity from other agents (local – only within VIEW_RANGE).
inline void get_gravity_from_agents_local(const float *__restrict all_locs, int skip_idx, int pow,
                                          float agent_x, float agent_y,
                                          float &out_gx, float &out_gy)
{
    float gx = 0.0f, gy = 0.0f;
    const int yc = static_cast<int>(agent_y);
    const int xc = static_cast<int>(agent_x);

    for (int j = 0; j < N_AGENTS; ++j)
    {
        if (j == skip_idx)
            continue;
        const float other_y = all_locs[j * 2];
        const float other_x = all_locs[j * 2 + 1];
        if (std::abs(static_cast<int>(other_y) - yc) > VIEW_RANGE ||
            std::abs(static_cast<int>(other_x) - xc) > VIEW_RANGE)
            continue;

        const float dx = other_x - agent_x;
        const float dy = other_y - agent_y;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq < DIST_SQ_MIN)
            continue;
        const float dist = std::sqrt(dist_sq);
        float denom;
        switch (pow)
        {
        case 1:
            denom = dist_sq;
            break;
        case 2:
            denom = dist * dist_sq;
            break;
        default:
        {
            denom = dist;
            for (int p = 0; p < pow; ++p)
                denom *= dist;
            break;
        }
        }
        const float f = 1.0f / denom;
        gx += dx * f;
        gy += dy * f;
    }
    out_gx = gx;
    out_gy = gy;
}

// Wall border force: virtual tiles of mass=1.0 at x=-1, x=MAP_SIZE, y=-1, y=MAP_SIZE.
inline void get_gravity_wall(int pow, float agent_x, float agent_y,
                             float &out_gx, float &out_gy, bool local)
{
    float gx = 0.0f, gy = 0.0f;

    auto accumulate_tile = [&](float wx, float wy)
    {
        const float dx = wx - agent_x;
        const float dy = wy - agent_y;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq < DIST_SQ_MIN)
            return;
        const float dist = std::sqrt(dist_sq);
        float denom;
        switch (pow)
        {
        case 1:
            denom = dist_sq;
            break;
        case 2:
            denom = dist * dist_sq;
            break;
        default:
        {
            denom = dist;
            for (int p = 0; p < pow; ++p)
                denom *= dist;
            break;
        }
        }
        const float f = 1.0f / denom;
        gx += dx * f;
        gy += dy * f;
    };

    if (local)
    {
        const int yc = static_cast<int>(agent_y);
        const int xc = static_cast<int>(agent_x);

        // Top wall (y = -1)
        if (yc + 1 <= VIEW_RANGE)
        {
            const int xs = std::max(-1, xc - VIEW_RANGE);
            const int xe = std::min(MAP_SIZE, xc + VIEW_RANGE) + 1;
            for (int x = xs; x < xe; ++x)
                accumulate_tile(static_cast<float>(x), -1.0f);
        }
        // Bottom wall (y = MAP_SIZE)
        if (MAP_SIZE - yc <= VIEW_RANGE)
        {
            const int xs = std::max(-1, xc - VIEW_RANGE);
            const int xe = std::min(MAP_SIZE, xc + VIEW_RANGE) + 1;
            for (int x = xs; x < xe; ++x)
                accumulate_tile(static_cast<float>(x), static_cast<float>(MAP_SIZE));
        }
        // Left wall (x = -1)
        if (xc + 1 <= VIEW_RANGE)
        {
            const int ys = std::max(-1, yc - VIEW_RANGE);
            const int ye = std::min(MAP_SIZE, yc + VIEW_RANGE) + 1;
            for (int y = ys; y < ye; ++y)
                accumulate_tile(-1.0f, static_cast<float>(y));
        }
        // Right wall (x = MAP_SIZE)
        if (MAP_SIZE - xc <= VIEW_RANGE)
        {
            const int ys = std::max(-1, yc - VIEW_RANGE);
            const int ye = std::min(MAP_SIZE, yc + VIEW_RANGE) + 1;
            for (int y = ys; y < ye; ++y)
                accumulate_tile(static_cast<float>(MAP_SIZE), static_cast<float>(y));
        }
    }
    else
    {
        // Global: all wall tiles
        for (int i = -1; i <= MAP_SIZE; ++i)
        {
            const float fi = static_cast<float>(i);
            accumulate_tile(fi, -1.0f);
            accumulate_tile(fi, static_cast<float>(MAP_SIZE));
            accumulate_tile(-1.0f, fi);
            accumulate_tile(static_cast<float>(MAP_SIZE), fi);
        }
    }

    out_gx = gx;
    out_gy = gy;
}

// Voronoi-partitioned undiscovered gravity (global).
// Only undiscovered tiles for which agent `agent_idx` is the nearest agent
// contribute.  This creates a territorial drive – each agent is pulled
// toward its own Voronoi cell of frontier tiles.
inline void get_gravity_voronoi(const float *__restrict disc_map,
                                const float *__restrict all_locs,
                                int agent_idx, int pow,
                                float agent_x, float agent_y,
                                float &out_gx, float &out_gy)
{
    float gx = 0.0f, gy = 0.0f;
    const float my_y = all_locs[agent_idx * 2];
    const float my_x = all_locs[agent_idx * 2 + 1];

    for (int y = 0; y < MAP_SIZE; ++y)
    {
        const float fy = static_cast<float>(y);
        const float dy_me = fy - my_y;
        const float *row = disc_map + y * MAP_SIZE;
        for (int x = 0; x < MAP_SIZE; ++x)
        {
            // Only undiscovered tiles (disc < 0.5 means not yet seen)
            if (row[x] > 0.5f)
                continue;

            const float fx = static_cast<float>(x);
            const float dx_me = fx - my_x;
            const float dist_sq_me = dx_me * dx_me + dy_me * dy_me;

            // Check if this agent is the closest
            bool closest = true;
            for (int j = 0; j < N_AGENTS; ++j)
            {
                if (j == agent_idx)
                    continue;
                const float dy_j = fy - all_locs[j * 2];
                const float dx_j = fx - all_locs[j * 2 + 1];
                if (dx_j * dx_j + dy_j * dy_j < dist_sq_me)
                {
                    closest = false;
                    break;
                }
            }
            if (!closest)
                continue;

            if (dist_sq_me < DIST_SQ_MIN)
                continue;
            const float dist = std::sqrt(dist_sq_me);
            float denom;
            switch (pow)
            {
            case 1:
                denom = dist_sq_me;
                break;
            case 2:
                denom = dist * dist_sq_me;
                break;
            default:
            {
                denom = dist;
                for (int p = 0; p < pow; ++p)
                    denom *= dist;
                break;
            }
            }
            const float f = 1.0f / denom;
            gx += dx_me * f;
            gy += dy_me * f;
        }
    }
    out_gx = gx;
    out_gy = gy;
}

// Voronoi-partitioned undiscovered gravity (local – VIEW_RANGE window).
inline void get_gravity_voronoi_local(const float *__restrict disc_map,
                                      const float *__restrict all_locs,
                                      int agent_idx, int pow,
                                      float agent_x, float agent_y,
                                      float &out_gx, float &out_gy)
{
    float gx = 0.0f, gy = 0.0f;
    const float my_y = all_locs[agent_idx * 2];
    const float my_x = all_locs[agent_idx * 2 + 1];
    const int yc = static_cast<int>(my_y);
    const int xc = static_cast<int>(my_x);
    const int y_s = std::max(0, yc - VIEW_RANGE);
    const int y_e = std::min(MAP_SIZE, yc + VIEW_RANGE + 1);
    const int x_s = std::max(0, xc - VIEW_RANGE);
    const int x_e = std::min(MAP_SIZE, xc + VIEW_RANGE + 1);

    for (int y = y_s; y < y_e; ++y)
    {
        const float fy = static_cast<float>(y);
        const float dy_me = fy - my_y;
        const float *row = disc_map + y * MAP_SIZE;
        for (int x = x_s; x < x_e; ++x)
        {
            if (row[x] > 0.5f)
                continue;

            const float fx = static_cast<float>(x);
            const float dx_me = fx - my_x;
            const float dist_sq_me = dx_me * dx_me + dy_me * dy_me;

            bool closest = true;
            for (int j = 0; j < N_AGENTS; ++j)
            {
                if (j == agent_idx)
                    continue;
                const float dy_j = fy - all_locs[j * 2];
                const float dx_j = fx - all_locs[j * 2 + 1];
                if (dx_j * dx_j + dy_j * dy_j < dist_sq_me)
                {
                    closest = false;
                    break;
                }
            }
            if (!closest)
                continue;

            if (dist_sq_me < DIST_SQ_MIN)
                continue;
            const float dist = std::sqrt(dist_sq_me);
            float denom;
            switch (pow)
            {
            case 1:
                denom = dist_sq_me;
                break;
            case 2:
                denom = dist * dist_sq_me;
                break;
            default:
            {
                denom = dist;
                for (int p = 0; p < pow; ++p)
                    denom *= dist;
                break;
            }
            }
            const float f = 1.0f / denom;
            gx += dx_me * f;
            gy += dy_me * f;
        }
    }
    out_gx = gx;
    out_gy = gy;
}

// Dispatch helper: map gravity, local or global.
inline void dispatch_map_gravity(const float *map, int pow,
                                 float agent_x, float agent_y,
                                 float &gx, float &gy,
                                 bool invert, bool local)
{
    if (local)
        get_gravity_local(map, pow, agent_x, agent_y, gx, gy, invert);
    else
        get_gravity(map, pow, agent_x, agent_y, gx, gy, invert);
}

// Shared procedural danger generator.
inline void generate_procedural_danger(float *danger, int e)
{
    const float ef = static_cast<float>(e);
    for (int i = 0; i < FLAT_MAP_SIZE; ++i)
    {
        const int y = i / MAP_SIZE;
        const int x = i % MAP_SIZE;
        float val = (std::sin(static_cast<float>(x) * 0.3f + ef) +
                     std::cos(static_cast<float>(y) * 0.3f + ef * 2.0f)) *
                    0.5f;
        danger[i] = std::fmin(1.0f, std::fmax(-1.0f, val));
    }
}

// Shared pybind11 FeatureType registration macro
#define REGISTER_FEATURE_TYPE_ENUM(m)                                          \
    py::enum_<FeatureType>(m, "FeatureType")                                   \
        .value("EXPECTED_DANGER", EXPECTED_DANGER_FEATURE)                     \
        .value("ACTUAL_DANGER", ACTUAL_DANGER_FEATURE)                         \
        .value("OBSERVED_DANGER", OBSERVED_DANGER_FEATURE)                     \
        .value("OBS", OBS_FEATURE)                                             \
        .value("EXPECTED_OBS", EXPECTED_OBS_FEATURE)                           \
        .value("GLOBAL_DISCOVERED", GLOBAL_DISCOVERED_FEATURE)                 \
        .value("OTHER_AGENTS", OTHER_AGENTS_FEATURE)                           \
        .value("OTHER_AGENTS_LAST_KNOWN", OTHER_AGENTS_LAST_KNOWN_FEATURE)     \
        .value("GLOBAL_UNDISCOVERED", GLOBAL_UNDISCOVERED_FEATURE)             \
        .value("OBS_UNDISCOVERED", OBS_UNDISCOVERED_FEATURE)                   \
        .value("EXPECTED_OBS_UNDISCOVERED", EXPECTED_OBS_UNDISCOVERED_FEATURE) \
        .value("RECENCY", RECENCY_FEATURE)                                     \
        .value("RECENCY_STALE", RECENCY_STALE_FEATURE)                         \
        .value("WALL_REPEL", WALL_REPEL_FEATURE)                               \
        .value("WALL_ATTRACT", WALL_ATTRACT_FEATURE)                           \
        .value("GLOBAL_VORONOI_UNDISCOVERED", GLOBAL_VORONOI_UNDISCOVERED_FEATURE) \
        .value("EXPECTED_VORONOI_UNDISCOVERED", EXPECTED_VORONOI_UNDISCOVERED_FEATURE)
