#include "poisson_disc.hpp"
#include "helper.hpp"

void poisson_distribution(int x, int y, std::unordered_map<std::pair<int, int>, PoissonChunk, PairHash>& grid) {
    PoissonChunk chunk;
    IntQueue<121> q;
    chunk.tiles
    grid[{x, y}] = chunk;
}