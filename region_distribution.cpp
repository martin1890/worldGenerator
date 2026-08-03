#include "region_distribution.hpp"

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{

constexpr int chunk_grid_side = 3;
constexpr int chunk_count = chunk_grid_side * chunk_grid_side;

constexpr int hash_width = 8;
constexpr int point_width = 16;

constexpr int padding_group_count = 8;
constexpr int padding_offset_count = padding_group_count + 1;

enum PaddingGroup : std::size_t
{
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Right,
    Bottom,
    Left
};

struct Bounds
{
    // Tile coordinates relative to the top-left chunk of the 3x3 area.
    int region_min_x;
    int region_min_y;
    int region_width;
    int region_height;

    int weight_min_x;
    int weight_min_y;
    int weight_width;
    int weight_height;
    int weight_stride;
};

struct WeightField
{
    std::vector<std::uint16_t> values;
    int stride;
};

struct ChunkBuilder
{
    RegionLocationsChunk chunk;

    std::array<
        std::vector<std::uint8_t>,
        padding_group_count
    > padding_groups;
};

inline int round_up(int value, int multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

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

    h = _mm256_xor_si256(h, _mm256_mullo_epi32(x, mx));
    h = _mm256_xor_si256(h, _mm256_mullo_epi32(y, my));
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
    h = _mm256_mullo_epi32(h, m1);
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 13));
    h = _mm256_mullo_epi32(h, m2);
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));

    return h;
}

inline __m128i truncate_u32x8_to_u16x8(__m256i values)
{
    values = _mm256_and_si256(
        values,
        _mm256_set1_epi32(0xFFFF));

    const __m128i low =
        _mm256_castsi256_si128(values);

    const __m128i high =
        _mm256_extracti128_si256(values, 1);

    return _mm_packus_epi32(low, high);
}

inline __m256i greater_u16(__m256i a, __m256i b)
{
    const __m256i sign =
        _mm256_set1_epi16(
            static_cast<short>(0x8000));

    return _mm256_cmpgt_epi16(
        _mm256_xor_si256(a, sign),
        _mm256_xor_si256(b, sign));
}

std::uint16_t find_missing_chunks(
    std::int32_t center_chunk_x,
    std::int32_t center_chunk_y,
    const std::unordered_map<
        std::pair<int, int>,
        RegionLocationsChunk,
        PairHash>& grid,
    int& min_chunk_x,
    int& min_chunk_y,
    int& max_chunk_x,
    int& max_chunk_y)
{
    std::uint16_t missing_mask = 0;

    min_chunk_x = chunk_grid_side;
    min_chunk_y = chunk_grid_side;
    max_chunk_x = -1;
    max_chunk_y = -1;

    for (int local_y = 0; local_y < chunk_grid_side; ++local_y) {
        for (int local_x = 0; local_x < chunk_grid_side; ++local_x) {
            const std::pair<int, int> position{
                center_chunk_x + local_x - 1,
                center_chunk_y + local_y - 1
            };

            if (grid.find(position) != grid.end()) {
                continue;
            }

            const int index =
                local_y * chunk_grid_side + local_x;

            missing_mask |=
                static_cast<std::uint16_t>(1u << index);

            if (local_x < min_chunk_x) {
                min_chunk_x = local_x;
            }

            if (local_x > max_chunk_x) {
                max_chunk_x = local_x;
            }

            if (local_y < min_chunk_y) {
                min_chunk_y = local_y;
            }

            if (local_y > max_chunk_y) {
                max_chunk_y = local_y;
            }
        }
    }

    return missing_mask;
}

Bounds calculate_bounds(
    int min_chunk_x,
    int min_chunk_y,
    int max_chunk_x,
    int max_chunk_y)
{
    /*
     * Region points are searched in the missing chunk rectangle plus
     * region_radius on every side.
     */
    const int region_min_x =
        min_chunk_x * chunk_side - region_radius;

    const int region_min_y =
        min_chunk_y * chunk_side - region_radius;

    const int region_max_x =
        (max_chunk_x + 1) * chunk_side + region_radius;

    const int region_max_y =
        (max_chunk_y + 1) * chunk_side + region_radius;

    /*
     * The weight field requires another region_radius around the
     * region-point search area.
     */
    const int weight_min_x =
        region_min_x - region_radius;

    const int weight_min_y =
        region_min_y - region_radius;

    const int weight_max_x =
        region_max_x + region_radius;

    const int weight_max_y =
        region_max_y + region_radius;

    const int weight_width =
        weight_max_x - weight_min_x;

    return {
        region_min_x,
        region_min_y,
        region_max_x - region_min_x,
        region_max_y - region_min_y,

        weight_min_x,
        weight_min_y,
        weight_width,
        weight_max_y - weight_min_y,
        round_up(weight_width, hash_width)
    };
}

void generate_weight_field(
    WeightField& field,
    const Bounds& bounds,
    std::uint32_t seed,
    std::int32_t center_chunk_x,
    std::int32_t center_chunk_y)
{
    field.stride = bounds.weight_stride;

    field.values.resize(
        static_cast<std::size_t>(bounds.weight_stride) *
        static_cast<std::size_t>(bounds.weight_height));

    const __m256i seed_vector =
        _mm256_set1_epi32(static_cast<int>(seed));

    const __m256i lane_offsets =
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    /*
     * Local coordinate zero is the beginning of the top-left chunk
     * in the 3x3 area.
     */
    const std::uint32_t area_world_x =
        static_cast<std::uint32_t>(center_chunk_x - 1) *
        static_cast<std::uint32_t>(chunk_side);

    const std::uint32_t area_world_y =
        static_cast<std::uint32_t>(center_chunk_y - 1) *
        static_cast<std::uint32_t>(chunk_side);

    const std::uint32_t start_world_x =
        area_world_x +
        static_cast<std::uint32_t>(bounds.weight_min_x);

    const std::uint32_t start_world_y =
        area_world_y +
        static_cast<std::uint32_t>(bounds.weight_min_y);

    for (int y = 0; y < bounds.weight_height; ++y) {
        const __m256i y_vector =
            _mm256_set1_epi32(static_cast<int>(
                start_world_y +
                static_cast<std::uint32_t>(y)));

        std::uint16_t* row =
            field.values.data() +
            static_cast<std::size_t>(y) * field.stride;

        for (int x = 0; x < field.stride; x += hash_width) {
            const __m256i x_vector =
                _mm256_add_epi32(
                    _mm256_set1_epi32(static_cast<int>(
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
                reinterpret_cast<__m128i*>(row + x),
                weights);
        }
    }
}

void add_padding_reference(
    ChunkBuilder& builder,
    std::uint8_t point_index,
    int point_x,
    int point_y)
{
    const bool left =
        point_x < region_radius;

    const bool right =
        point_x >= chunk_side - region_radius;

    const bool top =
        point_y < region_radius;

    const bool bottom =
        point_y >= chunk_side - region_radius;

    PaddingGroup group;

    if (top && left) {
        group = TopLeft;
    }
    else if (top && right) {
        group = TopRight;
    }
    else if (bottom && left) {
        group = BottomLeft;
    }
    else if (bottom && right) {
        group = BottomRight;
    }
    else if (top) {
        group = Top;
    }
    else if (right) {
        group = Right;
    }
    else if (bottom) {
        group = Bottom;
    }
    else if (left) {
        group = Left;
    }
    else {
        return;
    }

    builder.padding_groups[group].push_back(point_index);
}

void distribute_region_point(
    std::array<ChunkBuilder, chunk_count>& builders,
    std::uint16_t missing_mask,
    int area_x,
    int area_y)
{
    constexpr int complete_area_side =
        chunk_grid_side * chunk_side;

    /*
     * Points outside the actual 3x3 chunks belong to chunks outside
     * this operation. They are searched for halo purposes but are not
     * stored in these nine chunks.
     */
    if (area_x < 0 ||
        area_x >= complete_area_side ||
        area_y < 0 ||
        area_y >= complete_area_side) {
        return;
    }

    const int local_chunk_x =
        area_x / chunk_side;

    const int local_chunk_y =
        area_y / chunk_side;

    const int chunk_index =
        local_chunk_y * chunk_grid_side +
        local_chunk_x;

    if ((missing_mask & (1u << chunk_index)) == 0) {
        return;
    }

    const int point_x =
        area_x % chunk_side;

    const int point_y =
        area_y % chunk_side;

    ChunkBuilder& builder =
        builders[chunk_index];

    const auto point_index =
        static_cast<std::uint8_t>(
            builder.chunk.regions_x.size());

    builder.chunk.regions_x.push_back(
        static_cast<std::uint8_t>(point_x));

    builder.chunk.regions_y.push_back(
        static_cast<std::uint8_t>(point_y));

    add_padding_reference(
        builder,
        point_index,
        point_x,
        point_y);
}

void process_region_block(
    const WeightField& field,
    const Bounds& bounds,
    std::array<ChunkBuilder, chunk_count>& builders,
    std::uint16_t missing_mask,
    int region_x,
    int region_y,
    int first_lane_to_store)
{
    /*
     * Convert coordinates in the dynamic region rectangle to
     * coordinates inside the dynamic weight field.
     */
    const int field_x =
        region_x + region_radius;

    const int field_y =
        region_y + region_radius;

    const std::uint16_t* center_pointer =
        field.values.data() +
        static_cast<std::size_t>(field_y) * field.stride +
        field_x;

    const __m256i center =
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                center_pointer));

    __m256i mask =
        _mm256_set1_epi16(
            static_cast<short>(-1));

    for (int dy = -region_radius;
         dy <= region_radius;
         ++dy) {

        for (int dx = -region_radius;
             dx <= region_radius;
             ++dx) {

            if (dx == 0 && dy == 0) {
                continue;
            }

            const std::uint16_t* neighbor_pointer =
                field.values.data() +
                static_cast<std::size_t>(field_y + dy) *
                    field.stride +
                field_x + dx;

            const __m256i neighbor =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        neighbor_pointer));

            mask = _mm256_and_si256(
                mask,
                greater_u16(center, neighbor));

            if (_mm256_testz_si256(mask, mask)) {
                return;
            }
        }
    }

    const std::uint32_t bits =
        static_cast<std::uint32_t>(
            _mm256_movemask_epi8(mask));

    for (int lane = first_lane_to_store;
         lane < point_width;
         ++lane) {

        const std::uint32_t lane_mask =
            0x3u << (lane * 2);

        if ((bits & lane_mask) != lane_mask) {
            continue;
        }

        /*
         * Convert from the dynamic region rectangle back to coordinates
         * relative to the top-left chunk of the complete 3x3 area.
         */
        const int area_x =
            bounds.region_min_x + region_x + lane;

        const int area_y =
            bounds.region_min_y + region_y;

        distribute_region_point(
            builders,
            missing_mask,
            area_x,
            area_y);
    }
}

void find_region_points(
    const WeightField& field,
    const Bounds& bounds,
    std::array<ChunkBuilder, chunk_count>& builders,
    std::uint16_t missing_mask)
{
    const int remainder =
        bounds.region_width % point_width;

    const int complete_width =
        (bounds.region_width / point_width) *
        point_width;

    const int tail_start =
        bounds.region_width - point_width;

    const int first_tail_lane =
        point_width - remainder;

    for (int index = 0; index < chunk_count; ++index) {
        if ((missing_mask & (1u << index)) == 0) {
            continue;
        }

        builders[index].chunk.regions_x.reserve(64);
        builders[index].chunk.regions_y.reserve(64);
    }

    for (int y = 0; y < bounds.region_height; ++y) {
        for (int x = 0;
             x < complete_width;
             x += point_width) {

            process_region_block(
                field,
                bounds,
                builders,
                missing_mask,
                x,
                y,
                0);
        }

        if (remainder != 0) {
            process_region_block(
                field,
                bounds,
                builders,
                missing_mask,
                tail_start,
                y,
                first_tail_lane);
        }
    }
}

void pack_padding_indices(ChunkBuilder& builder)
{
    RegionLocationsChunk& chunk =
        builder.chunk;

    std::size_t reference_count = 0;

    for (const auto& group : builder.padding_groups) {
        reference_count += group.size();
    }

    chunk.padding_indices.clear();

    chunk.padding_indices.resize(
        padding_offset_count,
        0);

    chunk.padding_indices.reserve(
        padding_offset_count +
        reference_count);

    std::size_t offset = 0;

    for (std::size_t group = 0;
         group < padding_group_count;
         ++group) {

        chunk.padding_indices[group] =
            static_cast<std::uint8_t>(offset);

        for (std::uint8_t point_index :
             builder.padding_groups[group]) {

            chunk.padding_indices.push_back(point_index);
            ++offset;
        }
    }

    chunk.padding_indices[padding_group_count] =
        static_cast<std::uint8_t>(offset);
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
    int min_chunk_x;
    int min_chunk_y;
    int max_chunk_x;
    int max_chunk_y;

    const std::uint16_t missing_mask =
        find_missing_chunks(
            grid_x,
            grid_y,
            grid,
            min_chunk_x,
            min_chunk_y,
            max_chunk_x,
            max_chunk_y);

    if (missing_mask == 0) {
        return;
    }

    const Bounds bounds =
        calculate_bounds(
            min_chunk_x,
            min_chunk_y,
            max_chunk_x,
            max_chunk_y);

    WeightField weight_field;
    std::array<ChunkBuilder, chunk_count> builders;

    generate_weight_field(
        weight_field,
        bounds,
        seed,
        grid_x,
        grid_y);

    find_region_points(
        weight_field,
        bounds,
        builders,
        missing_mask);

    for (int local_y = 0; local_y < chunk_grid_side; ++local_y) {
        for (int local_x = 0;
             local_x < chunk_grid_side;
             ++local_x) {

            const int index =
                local_y * chunk_grid_side + local_x;

            if ((missing_mask & (1u << index)) == 0) {
                continue;
            }

            ChunkBuilder& builder =
                builders[index];

            pack_padding_indices(builder);

            const std::pair<int, int> position{
                grid_x + local_x - 1,
                grid_y + local_y - 1
            };

            grid.emplace(
                position,
                std::move(builder.chunk));
        }
    }
}
