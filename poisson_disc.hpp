#ifndef POISSON_DISC_HPP
#define POISSON_DISC_HPP

#include "helper.hpp"
#include <unordered_map>
#include <utility>
#include <functional>
#include <array>

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);

        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct PoissonChunk {
    std::array<float, 242> tiles{};
};

struct Points {
    float x;
    float y;
};

void poisson_distribution(std::unordered_map<std::pair<int, int>, PoissonChunk, PairHash>& grid);

#endif