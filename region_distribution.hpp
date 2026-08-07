#ifndef REGION_DISTRIBUTION_HPP
#define REGION_DISTRIBUTION_HPP

#include "helper.hpp"
#include <unordered_map>
#include <utility>
#include <functional>
#include <array>
#include <vector>

constexpr int region_length = 9;
constexpr int curve_reach = region_length - 1;       // 8
constexpr int chunk_side = 32;
constexpr int maxima_radius = 2;
constexpr int direction_radius = curve_reach;        // 8

// Points needed for direction calculations.
// 32 + 2*8 + 2*8 = 64
constexpr int region_points_side =
    chunk_side + curve_reach * 2 + direction_radius * 2;

// Extra maxima halo around the point area.
// 64 + 2*2 = 68
constexpr int weight_field_side =
    region_points_side + maxima_radius * 2;

// Eight uint16 hashes are generated at once.
constexpr int weight_field_stride =
    ((weight_field_side + 7) / 8) * 8;               // 72

constexpr int weight_field_size =
    weight_field_stride * weight_field_side;

constexpr int spatial_cell_side = direction_radius;  // 8
constexpr int spatial_grid_side =
    region_points_side / spatial_cell_side;          // 8

constexpr int spatial_cell_count =
    spatial_grid_side * spatial_grid_side;            // 64

constexpr uint8_t cell_capacity = 8;

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);

        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct RegionLocationsChunk {
    std::vector<std::uint8_t> regions_x;
    std::vector<std::uint8_t> regions_y;
    std::vector<std::uint16_t> region_weights;

    std::array<std::array<std::uint8_t, cell_capacity>, spatial_cell_count> spatial_cells_x{};
    std::array<std::array<std::uint8_t, cell_capacity>, spatial_cell_count> spatial_cells_y{};

    std::array<std::uint8_t, spatial_cell_count> spatial_cell_counts{};
    std::vector<std::uint8_t> overflow_cells;
};


void region_distribution(
    uint32_t seed,
    int32_t grid_x,
    int32_t grid_y,
    std::unordered_map<std::pair<int, int>, RegionLocationsChunk, PairHash>& grid);

#endif