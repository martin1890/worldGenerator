#include "region_distribution.hpp"

#include <immintrin.h>

namespace {

// Compares unsigned 16-bit integers using the signed SSE2 comparison.
// Flipping the top bit maps uint16_t ordering to int16_t ordering.
inline __m128i unsigned_greater_than_u16(__m128i a, __m128i b)
{
    const __m128i sign_bit = _mm_set1_epi16(static_cast<short>(0x8000));
    return _mm_cmpgt_epi16(_mm_xor_si128(a, sign_bit),
                           _mm_xor_si128(b, sign_bit));
}

bool is_local_max_scalar(
    const std::array<uint16_t, weight_field_size>& weight_field,
    int local_x,
    int local_y)
{
    const int center_x = local_x + region_radius;
    const int center_y = local_y + region_radius;
    const uint16_t center =
        weight_field[center_y * weight_field_side + center_x];

    for (int offset_y = -region_radius; offset_y <= region_radius; ++offset_y) {
        for (int offset_x = -region_radius; offset_x <= region_radius; ++offset_x) {
            if (offset_x == 0 && offset_y == 0) {
                continue;
            }

            const uint16_t neighbor = weight_field[
                (center_y + offset_y) * weight_field_side +
                (center_x + offset_x)];

            if (center <= neighbor) {
                return false;
            }
        }
    }

    return true;
}

} // namespace

void region_distribution(
    uint32_t seed,
    int32_t grid_x,
    int32_t grid_y,
    std::unordered_map<std::pair<int, int>, RegionLocationsChunk, PairHash>& grid)
{
    RegionLocationsChunk chunk;

    const int scaled_grid_x = grid_x * side;
    const int scaled_grid_y = grid_y * side;

    for (int y = 0; y < weight_field_side; ++y) {
        for (int x = 0; x < weight_field_side; ++x) {
            chunk.weight_field[y * weight_field_side + x] =
                hash_random_uint16_t(
                    seed,
                    scaled_grid_x + x - region_radius,
                    scaled_grid_y + y - region_radius);
        }
    }

    // A strict local maximum among (2 * region_radius + 1)^2 cells.
    // With region_radius == 2, every candidate is compared with 24 neighbors.
    constexpr int simd_width = 8;
    chunk.regions.reserve(tile_count / 4);

    for (int y = 0; y < side; ++y) {
        int x = 0;
        const int center_y = y + region_radius;

        // Eight uint16_t candidates are checked simultaneously.
        for (; x + simd_width <= side; x += simd_width) {
            const int center_x = x + region_radius;
            const uint16_t* center_ptr =
                chunk.weight_field.data() +
                center_y * weight_field_side + center_x;

            const __m128i center =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(center_ptr));
            __m128i local_max_mask = _mm_set1_epi16(-1);

            for (int offset_y = -region_radius;
                 offset_y <= region_radius;
                 ++offset_y) {
                for (int offset_x = -region_radius;
                     offset_x <= region_radius;
                     ++offset_x) {
                    if (offset_x == 0 && offset_y == 0) {
                        continue;
                    }

                    const uint16_t* neighbor_ptr =
                        chunk.weight_field.data() +
                        (center_y + offset_y) * weight_field_side +
                        (center_x + offset_x);

                    const __m128i neighbor = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(neighbor_ptr));

                    local_max_mask = _mm_and_si128(
                        local_max_mask,
                        unsigned_greater_than_u16(center, neighbor));
                }
            }

            // movemask gives two bits per uint16_t lane. A true lane has both set.
            unsigned int mask = static_cast<unsigned int>(
                _mm_movemask_epi8(local_max_mask));

            for (int lane = 0; lane < simd_width; ++lane) {
                const unsigned int lane_bits = 0x3u << (lane * 2);
                if ((mask & lane_bits) == lane_bits) {
                    chunk.regions.push_back({x + lane, y});
                }
            }
        }

        // Handles the remaining columns when side is not divisible by eight.
        for (; x < side; ++x) {
            if (is_local_max_scalar(chunk.weight_field, x, y)) {
                chunk.regions.push_back({x, y});
            }
        }
    }

    chunk.size = static_cast<int>(chunk.regions.size());
    grid[{grid_x, grid_y}] = std::move(chunk);
    std::printf(
        "Generated chunk (%d, %d) with %d regions.\n",
        grid_x,
        grid_y,
        static_cast<int>(grid[{grid_x, grid_y}].regions.size()));
}