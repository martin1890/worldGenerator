#ifndef REGION_DISTRIBUTION_HPP
#define REGION_DISTRIBUTION_HPP

#include "helper.hpp"
#include <unordered_map>
#include <utility>
#include <functional>
#include <array>

const int region_radius = 2;
const int side = 32;
const int weight_field_side = side + region_radius * 2;
const int tile_count = side * side;
const int weight_field_size = weight_field_side * weight_field_side;

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);

        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct Points {
    int x;
    int y;
};

struct RegionLocationsChunk {

    int size = tile_count;
    std::array<uint16_t, weight_field_size> weight_field{};
    std::vector<Points> regions;

};


void region_distribution(
    uint32_t seed,
    int32_t grid_x,
    int32_t grid_y,
    std::unordered_map<std::pair<int, int>, RegionLocationsChunk, PairHash>& grid);

#endif