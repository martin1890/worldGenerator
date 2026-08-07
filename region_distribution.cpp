#include "region_distribution.hpp"

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace
{

using WeightField =
    std::array<std::uint16_t, weight_field_size>;

constexpr int hash_width = 8;
constexpr int maxima_width = 16;

constexpr int point_area_offset =
    curve_reach + direction_radius; // 16

constexpr int weight_field_offset =
    point_area_offset + maxima_radius; // 18

static_assert(region_points_side == 64);
static_assert(weight_field_side == 68);
static_assert(weight_field_stride == 72);
static_assert(region_points_side % maxima_width == 0);
static_assert(weight_field_stride % hash_width == 0);
static_assert(spatial_grid_side == 8);
static_assert(cell_capacity == 8);


inline __m256i hash_u32x8(
    __m256i seed,
    __m256i x,
    __m256i y)
{
    const __m256i mx =
        _mm256_set1_epi32(static_cast<int>(374761393u));

    const __m256i my =
        _mm256_set1_epi32(static_cast<int>(668265263u));

    const __m256i m1 =
        _mm256_set1_epi32(static_cast<int>(2246822519u));

    const __m256i m2 =
        _mm256_set1_epi32(static_cast<int>(3266489917u));

    __m256i h = seed;

    h = _mm256_xor_si256(
        h,
        _mm256_mullo_epi32(x, mx));

    h = _mm256_xor_si256(
        h,
        _mm256_mullo_epi32(y, my));

    h = _mm256_xor_si256(
        h,
        _mm256_srli_epi32(h, 16));

    h = _mm256_mullo_epi32(h, m1);

    h = _mm256_xor_si256(
        h,
        _mm256_srli_epi32(h, 13));

    h = _mm256_mullo_epi32(h, m2);

    h = _mm256_xor_si256(
        h,
        _mm256_srli_epi32(h, 16));

    return h;
}


inline __m128i truncate_u32x8_to_u16x8(
    __m256i values)
{
    values = _mm256_and_si256(
        values,
        _mm256_set1_epi32(0xFFFF));

    const __m128i low =
        _mm256_castsi256_si128(values);

    const __m128i high =
        _mm256_extracti128_si256(values, 1);

    return _mm_packus_epi32(
        low,
        high);
}


inline __m256i greater_u16(
    __m256i a,
    __m256i b)
{
    const __m256i sign =
        _mm256_set1_epi16(
            static_cast<short>(0x8000));

    return _mm256_cmpgt_epi16(
        _mm256_xor_si256(a, sign),
        _mm256_xor_si256(b, sign));
}


void generate_weight_field(
    WeightField& field,
    std::uint32_t seed,
    std::int32_t grid_x,
    std::int32_t grid_y)
{
    const __m256i seed_vector =
        _mm256_set1_epi32(
            static_cast<int>(seed));

    const __m256i lane_offsets =
        _mm256_setr_epi32(
            0, 1, 2, 3,
            4, 5, 6, 7);

    /*
     * Use unsigned modulo arithmetic deliberately so negative
     * world coordinates behave deterministically.
     */
    const std::uint32_t chunk_world_x =
        static_cast<std::uint32_t>(grid_x) *
        static_cast<std::uint32_t>(chunk_side);

    const std::uint32_t chunk_world_y =
        static_cast<std::uint32_t>(grid_y) *
        static_cast<std::uint32_t>(chunk_side);

    const std::uint32_t start_world_x =
        chunk_world_x -
        static_cast<std::uint32_t>(weight_field_offset);

    const std::uint32_t start_world_y =
        chunk_world_y -
        static_cast<std::uint32_t>(weight_field_offset);

    for (int y = 0;
         y < weight_field_side;
         ++y) {

        const __m256i y_vector =
            _mm256_set1_epi32(
                static_cast<int>(
                    start_world_y +
                    static_cast<std::uint32_t>(y)));

        std::uint16_t* row =
            field.data() +
            static_cast<std::size_t>(y) *
                weight_field_stride;

        /*
         * 68 logical values are needed, but all 72 stored values
         * are generated so every row is pure SIMD.
         */
        for (int x = 0;
             x < weight_field_stride;
             x += hash_width) {

            const __m256i x_vector =
                _mm256_add_epi32(
                    _mm256_set1_epi32(
                        static_cast<int>(
                            start_world_x +
                            static_cast<std::uint32_t>(x))),
                    lane_offsets);

            const __m128i weights =
                truncate_u32x8_to_u16x8(
                    hash_u32x8(
                        seed_vector,
                        x_vector,
                        y_vector));

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(
                    row + x),
                weights);
        }
    }
}


void add_to_spatial_cell(
    RegionLocationsChunk& chunk,
    std::uint8_t x,
    std::uint8_t y)
{
    const std::uint8_t cell_x =
        x / spatial_cell_side;

    const std::uint8_t cell_y =
        y / spatial_cell_side;

    const std::uint8_t cell_index =
        static_cast<std::uint8_t>(
            cell_y * spatial_grid_side +
            cell_x);

    std::uint8_t& count =
        chunk.spatial_cell_counts[cell_index];

    if (count < cell_capacity) {
        chunk.spatial_cells_x[cell_index][count] = x;
        chunk.spatial_cells_y[cell_index][count] = y;
        ++count;
        return;
    }

    /*
     * Only record the cell once.
     *
     * With maxima_radius == 2, nine points is already the
     * theoretical extreme for an 8x8 cell.
     */
    if (count == cell_capacity) {
        chunk.overflow_cells.push_back(cell_index);

        std::printf(
            "Spatial cell overflow in cell %u\n",
            static_cast<unsigned>(cell_index));

        /*
         * 9 means that an overflow has occurred while the first
         * eight entries remain the usable SIMD set.
         */
        ++count;
    }
}


void find_region_points(
    const WeightField& field,
    RegionLocationsChunk& chunk)
{
    /*
     * Rough expected count:
     * 4096 / 25 ~= 164.
     */
    chunk.regions_x.reserve(192);
    chunk.regions_y.reserve(192);
    chunk.region_weights.reserve(192);

    chunk.overflow_cells.reserve(4);

    /*
     * The 64x64 point area begins two cells into the 68x68
     * logical weight field.
     *
     * Because 64 is divisible by 16, every candidate is handled
     * by AVX2 with no scalar or overlapping tail.
     */
    for (int y = 0;
         y < region_points_side;
         ++y) {

        const int field_y =
            y + maxima_radius;

        for (int x = 0;
             x < region_points_side;
             x += maxima_width) {

            const int field_x =
                x + maxima_radius;

            const std::uint16_t* center_pointer =
                field.data() +
                static_cast<std::size_t>(field_y) *
                    weight_field_stride +
                field_x;

            const __m256i center =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        center_pointer));

            __m256i mask =
                _mm256_set1_epi16(
                    static_cast<short>(-1));

            for (int dy = -maxima_radius;
                 dy <= maxima_radius;
                 ++dy) {

                for (int dx = -maxima_radius;
                     dx <= maxima_radius;
                     ++dx) {

                    if (dx == 0 && dy == 0) {
                        continue;
                    }

                    const std::uint16_t* neighbor_pointer =
                        field.data() +
                        static_cast<std::size_t>(
                            field_y + dy) *
                            weight_field_stride +
                        field_x + dx;

                    const __m256i neighbor =
                        _mm256_loadu_si256(
                            reinterpret_cast<
                                const __m256i*>(
                                neighbor_pointer));

                    mask = _mm256_and_si256(
                        mask,
                        greater_u16(
                            center,
                            neighbor));

                    if (_mm256_testz_si256(
                            mask,
                            mask)) {
                        break;
                    }
                }

                if (_mm256_testz_si256(
                        mask,
                        mask)) {
                    break;
                }
            }

            const std::uint32_t bits =
                static_cast<std::uint32_t>(
                    _mm256_movemask_epi8(mask));

            if (bits == 0) {
                continue;
            }

            /*
             * Extract the sparse maxima from the SIMD result.
             * This bookkeeping is intentionally scalar.
             */
            for (int lane = 0;
                 lane < maxima_width;
                 ++lane) {

                const std::uint32_t lane_mask =
                    0x3u << (lane * 2);

                if ((bits & lane_mask) != lane_mask) {
                    continue;
                }

                const std::uint8_t point_x =
                    static_cast<std::uint8_t>(
                        x + lane);

                const std::uint8_t point_y =
                    static_cast<std::uint8_t>(y);

                chunk.regions_x.push_back(point_x);
                chunk.regions_y.push_back(point_y);
                chunk.region_weights.push_back(
                    center_pointer[lane]);

                add_to_spatial_cell(
                    chunk,
                    point_x,
                    point_y);
            }
        }
    }
}

} // namespace


void region_distribution(
    std::uint32_t seed,
    std::int32_t grid_x,
    std::int32_t grid_y,
    std::unordered_map<
        std::pair<int, int>,
        RegionLocationsChunk,
        PairHash>& grid)
{
    const std::pair<int, int> position{
        grid_x,
        grid_y
    };

    if (grid.find(position) != grid.end()) {
        return;
    }

    WeightField weight_field;
    RegionLocationsChunk chunk;

    generate_weight_field(
        weight_field,
        seed,
        grid_x,
        grid_y);

    find_region_points(
        weight_field,
        chunk);

    grid.emplace(
        position,
        std::move(chunk));
}