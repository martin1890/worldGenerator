#ifndef REGION_DISTRIBUTION_HPP
#define REGION_DISTRIBUTION_HPP

#include "helper.hpp"
#include <unordered_map>
#include <utility>
#include <functional>
#include <array>
#include <vector>

constexpr int region_length = 9;
constexpr int curve_reach = region_length - 1;  // 8

constexpr int chunk_side = 32;
constexpr int maxima_radius = 2;

constexpr int direction_radius = curve_reach;   // 8
constexpr int spatial_cell_side = direction_radius; // 8


constexpr int chunk_cell_side =
    chunk_side / spatial_cell_side;             // 4

constexpr int curve_support_halo_cells = 2;
constexpr int direction_halo_cells = 1;


constexpr int spatial_grid_side =
    chunk_cell_side +
    2 * curve_support_halo_cells +
    2 * direction_halo_cells;                   // 10

constexpr int spatial_cell_count =
    spatial_grid_side * spatial_grid_side;       // 100


constexpr int region_points_side =
    spatial_grid_side * spatial_cell_side;       // 80


constexpr int point_area_offset =
    (direction_halo_cells +
     curve_support_halo_cells) *
    spatial_cell_side;                           // 24


constexpr int weight_field_side =
    region_points_side +
    maxima_radius * 2;                           // 84

constexpr int weight_field_stride =
    ((weight_field_side + 7) / 8) * 8;           // 88

constexpr int weight_field_size =
    weight_field_stride *
    weight_field_side;


constexpr uint8_t cell_capacity = 8;
constexpr int curve_step_count = region_length - 1; // 8
constexpr int curve_start_shift = 24;
constexpr int curve_point_count_shift = 28;
constexpr std::uint32_t curve_steps_mask = 0x00FFFFFFu;

constexpr std::array<int, 8> curve_dx = {
     1,  1,  0, -1,
    -1, -1,  0,  1
};

constexpr std::array<int, 8> curve_dy = {
     0,  1,  1,  1,
     0, -1, -1, -1
};

inline std::uint8_t get_curve_start(
    std::uint32_t curve)
{
    return static_cast<std::uint8_t>(
        (curve >> curve_start_shift) & 0x0Fu);
}

inline std::uint8_t get_curve_point_count(
    std::uint32_t curve)
{
    return static_cast<std::uint8_t>(
        (curve >> curve_point_count_shift) & 0x0Fu);
}

inline std::uint8_t get_curve_step(
    std::uint32_t curve,
    int step)
{
    return static_cast<std::uint8_t>(
        (curve >> (step * 3)) & 0x7u);
}

inline void set_curve_start(
    std::uint32_t& curve,
    std::uint8_t start)
{
    curve =
        (curve & ~(0x0Fu << curve_start_shift)) |
        ((static_cast<std::uint32_t>(start) & 0x0Fu)
            << curve_start_shift);
}

inline void set_curve_point_count(
    std::uint32_t& curve,
    std::uint8_t point_count)
{
    curve =
        (curve & ~(0x0Fu << curve_point_count_shift)) |
        ((static_cast<std::uint32_t>(point_count) & 0x0Fu)
            << curve_point_count_shift);
}

static_assert(curve_step_count == 8);

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);

        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct RegionLocationsChunk {

    enum class ChunkGenerationState : std::uint8_t {
        Empty,
        CurvesPrepared,
        Finished
    };

    ChunkGenerationState generation_state = ChunkGenerationState::Empty;
    std::uint16_t received_neighbor_mask = 0;

    std::array<
        std::array<std::uint8_t, spatial_grid_side>,
        spatial_grid_side>
        cell_generation_state{};

    /*
    * Cell generation states:
    * 0 = empty
    * 1 = points available
    * 2 = points + directions available
    * 3 = points + directions + curves available
    */

    std::vector<std::uint8_t> regions_x;
    std::vector<std::uint8_t> regions_y;
    std::vector<std::uint16_t> region_weights;
    std::vector<float> region_directions;
    std::vector<std::uint32_t> region_curves;

    std::vector<std::uint8_t> curve_min_x;
    std::vector<std::uint8_t> curve_max_x;
    std::vector<std::uint8_t> curve_min_y;
    std::vector<std::uint8_t> curve_max_y;

    std::vector<std::uint16_t> region_curve_indices;
    std::vector<std::array<std::uint16_t, 16>> temporary_curves;

    std::array<std::array<std::uint8_t, cell_capacity>, spatial_cell_count> spatial_cells_x{};
    std::array<std::array<std::uint8_t, cell_capacity>, spatial_cell_count> spatial_cells_y{};
    std::array<std::array<std::uint16_t, cell_capacity>, spatial_cell_count> spatial_cells_indices{};

    std::array<std::uint8_t, spatial_cell_count> spatial_cell_counts{};
    std::vector<std::uint8_t> overflow_cells;
};


void region_distribution(
    uint32_t seed,
    int32_t grid_x,
    int32_t grid_y,
    std::unordered_map<std::pair<int, int>, RegionLocationsChunk, PairHash>& grid);

#endif