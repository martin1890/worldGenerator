#ifndef REGION_DISTRIBUTION_HPP
#define REGION_DISTRIBUTION_HPP

#include "helper.hpp"
#include <unordered_map>
#include <utility>
#include <functional>
#include <array>
#include <vector>

constexpr int region_radius = 2;
constexpr int chunk_side = 32;

constexpr int chunks_area_side =
    chunk_side * 3 + region_radius * 2;                  // 100

constexpr int weight_field_side =
    chunks_area_side + region_radius * 2;                   // 104

constexpr int weight_field_padded =
    ((weight_field_side + 7) / 8) * 8;                      // 104

constexpr int weight_field_size =
    weight_field_padded * weight_field_side;                // 10816

constexpr int region_length = 8;

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);

        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct RegionLocationsChunk {
    std::vector<uint8_t> regions_x;
    std::vector<uint8_t> regions_y;

    // [9 offsets][point indices]
    std::vector<std::uint8_t> padding_indices;
};


void region_distribution(
    uint32_t seed,
    int32_t grid_x,
    int32_t grid_y,
    std::unordered_map<std::pair<int, int>, RegionLocationsChunk, PairHash>& grid);

#endif