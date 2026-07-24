#include "poisson_disc.hpp"
#include "helper.hpp"

void poisson_distribution(uint32_t seed, int32_t x, int32_t y, std::unordered_map<std::pair<int, int>, PoissonChunk, PairHash>& grid) {

    const int tries = 20;
    float rng = hash_random(seed, x, y);


    PoissonChunk chunk;
    IntQueue<tile_count> q;
    q.push(tile_count / 2); // Start from the middle of the chunk

    while (!q.empty()) {
        active_checklist = 
        for (int i = tries; i > 0; i--) {
            chunk.tiles[q.front()] = {new_random(rng), new_random(rng)};
        }
        chunk.tiles[]
    }
    grid[{x, y}] = chunk;
}