#include "region_distribution.hpp"

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <algorithm>
#include <cmath>

namespace
{

using WeightField =
    std::array<std::uint16_t, weight_field_size>;

constexpr int hash_width = 8;
constexpr int maxima_width = 16;

constexpr int weight_field_offset =
    point_area_offset + maxima_radius; // 18

constexpr int curve_support_min_cell =
    direction_halo_cells;                       // 1

constexpr int curve_support_max_cell =
    spatial_grid_side -
    direction_halo_cells - 1;                    // 8


constexpr int chunk_curve_min_cell =
    direction_halo_cells +
    curve_support_halo_cells;                    // 3

constexpr int chunk_curve_max_cell =
    chunk_curve_min_cell +
    chunk_cell_side - 1;                         // 6

constexpr int curve_collision_radius = 1;

static_assert(region_points_side == 80);
static_assert(weight_field_side == 84);
static_assert(weight_field_stride == 88);
static_assert(region_points_side % maxima_width == 0);
static_assert(weight_field_stride % hash_width == 0);
static_assert(spatial_grid_side == 10);
static_assert(cell_capacity == 8);
static_assert(chunk_cell_side == 4);
static_assert(point_area_offset == 24);
static_assert(curve_support_min_cell == 1);
static_assert(curve_support_max_cell == 8);
static_assert(chunk_curve_min_cell == 3);
static_assert(chunk_curve_max_cell == 6);

static constexpr float markov_matrix[5][5] = {
    {0.10, 0.90, 0.00, 0.00, 0.00},
    {0.20, 0.30, 0.50, 0.00, 0.00},
    {0.00, 0.35, 0.30, 0.35, 0.00},
    {0.00, 0.00, 0.50, 0.30, 0.20},
    {0.00, 0.00, 0.00, 0.90, 0.10}
};

constexpr std::array<float, 8> grid_dir_x = {
     1.0f,
     0.70710678118f,
     0.0f,
    -0.70710678118f,
    -1.0f,
    -0.70710678118f,
     0.0f,
     0.70710678118f
};

constexpr std::array<float, 8> grid_dir_y = {
     0.0f,
     0.70710678118f,
     1.0f,
     0.70710678118f,
     0.0f,
    -0.70710678118f,
    -1.0f,
    -0.70710678118f
};


std::uint8_t delta_to_curve_direction(
    int dx,
    int dy)
{
    if (dx ==  1 && dy ==  0) return 0;
    if (dx ==  1 && dy ==  1) return 1;
    if (dx ==  0 && dy ==  1) return 2;
    if (dx == -1 && dy ==  1) return 3;
    if (dx == -1 && dy ==  0) return 4;
    if (dx == -1 && dy == -1) return 5;
    if (dx ==  0 && dy == -1) return 6;
    return 7;
}


constexpr float pi = 3.14159265358979323846f;

constexpr float quarter_turn =
    pi * 0.5f;

constexpr float grid_angle_step =
    pi * 0.25f;




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

struct CurveBounds {
    int min_x;
    int max_x;
    int min_y;
    int max_y;
};

inline bool bounds_overlap_with_margin(
    const CurveBounds& a,
    const CurveBounds& b,
    int margin)
{
    return
        a.min_x <= b.max_x + margin &&
        a.max_x + margin >= b.min_x &&
        a.min_y <= b.max_y + margin &&
        a.max_y + margin >= b.min_y;
}

inline bool bounds_overlap(
    const CurveBounds& a,
    const CurveBounds& b)
{
    return
        a.min_x <= b.max_x &&
        a.max_x >= b.min_x &&
        a.min_y <= b.max_y &&
        a.max_y >= b.min_y;
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
    std::uint8_t y,
    std::uint16_t region_index)
{
    const int cell_x =
        x / spatial_cell_side;

    const int cell_y =
        y / spatial_cell_side;

    const int cell_index =
        cell_y * spatial_grid_side +
        cell_x;

    std::uint8_t& count =
        chunk.spatial_cell_counts[cell_index];

    if (count < cell_capacity) {
        chunk.spatial_cells_x[cell_index][count] = x;
        chunk.spatial_cells_y[cell_index][count] = y;

        chunk.spatial_cells_indices[cell_index][count] =
            region_index;

        ++count;
        return;
    }

    /*
     * For now, ignore points beyond the first eight in the
     * spatial cell. The actual region still exists in the
     * region vectors.
     */
    chunk.overflow_cells.push_back(
        static_cast<std::uint8_t>(cell_index));

    std::printf(
        "Spatial cell overflow in cell %d\n",
        cell_index);
}


void find_region_points(
    RegionLocationsChunk& chunk,
    std::uint32_t seed,
    std::int32_t grid_x,
    std::int32_t grid_y)
{
    /*
     * Rough expected count:
     * 4096 / 25 ~= 164.
     */

    WeightField weight_field;

    generate_weight_field(
        weight_field,
        seed,
        grid_x,
        grid_y);

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
                weight_field.data() +
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
                        weight_field.data() +
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

                /*
                * The new point will get this index in all parallel
                * region vectors.
                */
                const std::uint16_t region_index =
                    static_cast<std::uint16_t>(
                        chunk.regions_x.size());

                chunk.regions_x.push_back(point_x);
                chunk.regions_y.push_back(point_y);
                chunk.region_weights.push_back(
                    center_pointer[lane]);

                add_to_spatial_cell(
                    chunk,
                    point_x,
                    point_y,
                    region_index);
            }
        }
    }
}

} // namespace


inline float horizontal_sum(__m256 v)
{
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);

    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}

void calculate_region_directions(
    RegionLocationsChunk& chunk,
    std::uint32_t seed,
    std::int32_t grid_x,
    std::int32_t grid_y)
{
    const std::size_t point_count = chunk.regions_x.size();

    chunk.region_directions.resize(point_count);

    const __m256i lane_indices =
        _mm256_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7);

    const __m256i max_distance_squared =
        _mm256_set1_epi32(
            direction_radius * direction_radius);


    // Only cells that produce curves need directions.
    for (int cell_y = curve_support_min_cell;
         cell_y <= curve_support_max_cell;
         ++cell_y) {

        for (int cell_x = curve_support_min_cell;
             cell_x <= curve_support_max_cell;
             ++cell_x) {

            const int cell_index =
                cell_y * spatial_grid_side + cell_x;

            const int cell_count =
                chunk.spatial_cell_counts[cell_index];


            for (int slot = 0; slot < cell_count; ++slot) {

                const std::uint16_t point_index =
                    chunk.spatial_cells_indices[cell_index][slot];

                const int point_x =
                    chunk.regions_x[point_index];

                const int point_y =
                    chunk.regions_y[point_index];

                float repulsion_x = 0.0f;
                float repulsion_y = 0.0f;


                // Because the current cell is always in 1..8,
                // all neighboring cells are guaranteed to be in 0..9.
                for (int cell_offset_y = -1;
                     cell_offset_y <= 1;
                     ++cell_offset_y) {

                    const int neighbor_cell_y =
                        cell_y + cell_offset_y;

                    for (int cell_offset_x = -1;
                         cell_offset_x <= 1;
                         ++cell_offset_x) {

                        const int neighbor_cell_x =
                            cell_x + cell_offset_x;

                        const int neighbor_cell_index =
                            neighbor_cell_y * spatial_grid_side +
                            neighbor_cell_x;

                        const int count =
                            chunk.spatial_cell_counts[
                                neighbor_cell_index];

                        if (count == 0) {
                            continue;
                        }


                        const __m128i packed_x =
                            _mm_loadl_epi64(
                                reinterpret_cast<const __m128i*>(
                                    chunk.spatial_cells_x[
                                        neighbor_cell_index].data()));

                        const __m128i packed_y =
                            _mm_loadl_epi64(
                                reinterpret_cast<const __m128i*>(
                                    chunk.spatial_cells_y[
                                        neighbor_cell_index].data()));


                        const __m256i neighbor_x =
                            _mm256_cvtepu8_epi32(
                                packed_x);

                        const __m256i neighbor_y =
                            _mm256_cvtepu8_epi32(
                                packed_y);


                        const __m256i dx =
                            _mm256_sub_epi32(
                                _mm256_set1_epi32(point_x),
                                neighbor_x);

                        const __m256i dy =
                            _mm256_sub_epi32(
                                _mm256_set1_epi32(point_y),
                                neighbor_y);


                        const __m256i distance_squared =
                            _mm256_add_epi32(
                                _mm256_mullo_epi32(dx, dx),
                                _mm256_mullo_epi32(dy, dy));


                        const __m256i valid_count =
                            _mm256_cmpgt_epi32(
                                _mm256_set1_epi32(count),
                                lane_indices);


                        // Exclude the point itself.
                        const __m256i nonzero =
                            _mm256_cmpgt_epi32(
                                distance_squared,
                                _mm256_setzero_si256());


                        const __m256i within_radius =
                            _mm256_xor_si256(
                                _mm256_cmpgt_epi32(
                                    distance_squared,
                                    max_distance_squared),
                                _mm256_set1_epi32(-1));


                        const __m256i valid =
                            _mm256_and_si256(
                                valid_count,
                                _mm256_and_si256(
                                    nonzero,
                                    within_radius));


                        const __m256 distance_float =
                            _mm256_cvtepi32_ps(
                                distance_squared);


                        const __m256 inverse_distance =
                            _mm256_div_ps(
                                _mm256_set1_ps(1.0f),
                                distance_float);


                        const __m256 weight =
                            _mm256_and_ps(
                                inverse_distance,
                                _mm256_castsi256_ps(valid));


                        const __m256 contribution_x =
                            _mm256_mul_ps(
                                _mm256_cvtepi32_ps(dx),
                                weight);

                        const __m256 contribution_y =
                            _mm256_mul_ps(
                                _mm256_cvtepi32_ps(dy),
                                weight);


                        repulsion_x +=
                            horizontal_sum(contribution_x);

                        repulsion_y +=
                            horizontal_sum(contribution_y);
                    }
                }


                const std::int32_t world_x =
                    grid_x * chunk_side +
                    point_x -
                    point_area_offset;

                const std::int32_t world_y =
                    grid_y * chunk_side +
                    point_y -
                    point_area_offset;


                const float random_angle =
                    hash_random(
                        seed ^ 0xA511E9B3u,
                        world_x,
                        world_y)
                    * 2.0f * pi;


                const float random_x =
                    std::cos(random_angle);

                const float random_y =
                    std::sin(random_angle);


                constexpr float repulsion_strength = 1.5f;


                const float direction_x =
                    random_x +
                    repulsion_x * repulsion_strength;

                const float direction_y =
                    random_y +
                    repulsion_y * repulsion_strength;


                float direction =
                    std::atan2(
                        direction_y,
                        direction_x);


                if (direction < 0.0f) {
                    direction += 2.0f * pi;
                }


                chunk.region_directions[point_index] =
                    direction;
            }
        }
    }
}


int sample_next_state(
    int current_state,
    float random_value)
{
    float cumulative = 0.0f;

    for (int next_state = 0;
         next_state < 5;
         ++next_state) {

        cumulative +=
            markov_matrix[current_state][next_state];

        if (random_value < cumulative) {
            return next_state;
        }
    }

    return 4;
}


std::uint8_t angle_to_grid_direction(
    float angle,
    float main_angle,
    float random_value)
{
    constexpr float full_turn =
        2.0f * pi;

    while (angle < 0.0f) {
        angle += full_turn;
    }

    while (angle >= full_turn) {
        angle -= full_turn;
    }

    const float grid_position =
        angle / grid_angle_step;

    const int lower_direction =
        static_cast<int>(
            std::floor(grid_position));

    const int upper_direction =
        (lower_direction + 1) & 7;

    const float upper_probability =
        grid_position -
        static_cast<float>(lower_direction);


    const float main_x =
        std::cos(main_angle);

    const float main_y =
        std::sin(main_angle);


    const float lower_dot =
        grid_dir_x[lower_direction] * main_x +
        grid_dir_y[lower_direction] * main_y;

    const float upper_dot =
        grid_dir_x[upper_direction] * main_x +
        grid_dir_y[upper_direction] * main_y;


    const bool lower_valid =
        lower_dot >= 0.0f;

    const bool upper_valid =
        upper_dot >= 0.0f;


    /*
     * If only one candidate is allowed, use it directly.
     */
    if (lower_valid && !upper_valid) {
        return static_cast<std::uint8_t>(
            lower_direction);
    }

    if (!lower_valid && upper_valid) {
        return static_cast<std::uint8_t>(
            upper_direction);
    }


    /*
     * Normally both are valid.
     */
    if (lower_valid && upper_valid) {
        if (random_value < upper_probability) {
            return static_cast<std::uint8_t>(
                upper_direction);
        }

        return static_cast<std::uint8_t>(
            lower_direction);
    }


    /*
     * Safety fallback.
     *
     * This should not normally occur because desired_angle is
     * always within +-90 degrees of main_angle.
     */
    return static_cast<std::uint8_t>(
        lower_dot > upper_dot
            ? lower_direction
            : upper_direction);
}


void create_region_curves(
    RegionLocationsChunk& chunk,
    std::uint32_t seed,
    std::int32_t grid_x,
    std::int32_t grid_y,

    std::vector<uint8_t>& curve_min_x,
    std::vector<uint8_t>& curve_max_x,
    std::vector<uint8_t>& curve_min_y,
    std::vector<uint8_t>& curve_max_y,
    std::vector<std::array<std::uint16_t, 16>>& temporary_curves,
    std::vector<std::uint16_t>& curve_region_indices,
    std::vector<std::uint16_t>& region_curve_indices) 
{
    /*
     * Keep region_curves parallel with the other region vectors.
     * Points outside the curve-producing area remain zero.
     */
    const std::size_t point_count = chunk.regions_x.size();

    curve_min_x.reserve(point_count);
    curve_max_x.reserve(point_count);
    curve_min_y.reserve(point_count);
    curve_max_y.reserve(point_count);
    temporary_curves.reserve(point_count);
    curve_region_indices.reserve(point_count);

    for (int cell_y = curve_support_min_cell;
         cell_y <= curve_support_max_cell;
         ++cell_y) {

        for (int cell_x = curve_support_min_cell;
             cell_x <= curve_support_max_cell;
             ++cell_x) {

            const int cell_index =
                cell_y * spatial_grid_side +
                cell_x;

            const int count =
                chunk.spatial_cell_counts[
                    cell_index];


            for (int slot = 0;
                 slot < count;
                 ++slot) {

                const std::uint16_t point_index =
                    chunk.spatial_cells_indices[
                        cell_index][slot];

                const std::uint8_t point_x =
                    chunk.regions_x[point_index];

                const std::uint8_t point_y =
                    chunk.regions_y[point_index];


                std::uint8_t curve_x = point_x;
                std::uint8_t curve_y = point_y;

                std::uint8_t min_x = curve_x;
                std::uint8_t max_x = curve_x;
                std::uint8_t min_y = curve_y;
                std::uint8_t max_y = curve_y;


                std::array<std::uint16_t, 16>
                    temporary_curve;

                temporary_curve.fill(0xFFFFu);

                /*
                 * Position 0 is the curve origin.
                 */
                temporary_curve[0] =
                    (static_cast<std::uint16_t>(curve_x) << 8) |
                    static_cast<std::uint16_t>(curve_y);


                const std::int32_t world_x =
                    grid_x * chunk_side +
                    point_x -
                    point_area_offset;

                const std::int32_t world_y =
                    grid_y * chunk_side +
                    point_y -
                    point_area_offset;


                float random_value =
                    hash_random(
                        seed ^ 0x6C8E9CF5u,
                        world_x,
                        world_y);

                const float main_angle =
                    chunk.region_directions[
                        point_index];

                int state = 2;


                for (int step = 0;
                     step < curve_step_count;
                     ++step) {

                    const float state_offset =
                        static_cast<float>(
                            state - 2) *
                        grid_angle_step;

                    const float desired_angle =
                        main_angle +
                        state_offset;


                    random_value =
                        new_random(
                            random_value);

                    const std::uint8_t grid_direction =
                        angle_to_grid_direction(
                            desired_angle,
                            main_angle,
                            random_value);


                    curve_x +=
                        curve_dx[grid_direction];

                    curve_y +=
                        curve_dy[grid_direction];


                    min_x =
                        std::min(
                            min_x,
                            curve_x);

                    max_x =
                        std::max(
                            max_x,
                            curve_x);

                    min_y =
                        std::min(
                            min_y,
                            curve_y);

                    max_y =
                        std::max(
                            max_y,
                            curve_y);


                    /*
                     * step 0 becomes position 1.
                     * step 7 becomes position 8.
                     */
                    temporary_curve[step + 1] =
                        (static_cast<std::uint16_t>(curve_x) << 8) |
                        static_cast<std::uint16_t>(curve_y);


                    random_value =
                        new_random(
                            random_value);

                    state =
                        sample_next_state(
                            state,
                            random_value);
                }
                const std::uint16_t curve_index = static_cast<std::uint16_t>(temporary_curves.size());

                region_curve_indices[point_index] = curve_index;

                curve_region_indices.push_back(
                    point_index);

                curve_min_x.push_back(
                    min_x);

                curve_max_x.push_back(
                    max_x);

                curve_min_y.push_back(
                    min_y);

                curve_max_y.push_back(
                    max_y);

                temporary_curves.push_back(
                    temporary_curve);
            }
        }
    }


    if (!chunk.overflow_cells.empty()) {
        /*
         * TODO: Handle curve generation for overflow points.
         */
    }
}


std::uint16_t get_collision_mask(
    const std::array<std::uint16_t, 16>& current_curve,
    const std::array<std::uint16_t, 16>& other_curve)
{
    const __m256i current_tiles =
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                current_curve.data()));

    const __m256i current_x =
        _mm256_srli_epi16(
            current_tiles,
            8);

    const __m256i current_y =
        _mm256_and_si256(
            current_tiles,
            _mm256_set1_epi16(0x00FF));

    const __m256i two =
        _mm256_set1_epi16(2);

    __m256i matches =
        _mm256_setzero_si256();


    for (int other_position = 0;
         other_position < region_length;
         ++other_position) {

        const std::uint16_t other_tile =
            other_curve[other_position];

        const int other_x =
            static_cast<int>(
                other_tile >> 8);

        const int other_y =
            static_cast<int>(
                other_tile & 0xFFu);


        const __m256i dx =
            _mm256_abs_epi16(
                _mm256_sub_epi16(
                    current_x,
                    _mm256_set1_epi16(
                        static_cast<short>(other_x))));

        const __m256i dy =
            _mm256_abs_epi16(
                _mm256_sub_epi16(
                    current_y,
                    _mm256_set1_epi16(
                        static_cast<short>(other_y))));


        // 2 > distance means distance <= 1.
        const __m256i close_x =
            _mm256_cmpgt_epi16(
                two,
                dx);

        const __m256i close_y =
            _mm256_cmpgt_epi16(
                two,
                dy);


        const __m256i close =
            _mm256_and_si256(
                close_x,
                close_y);

        matches =
            _mm256_or_si256(
                matches,
                close);
    }


    const std::uint32_t bits =
        static_cast<std::uint32_t>(
            _mm256_movemask_epi8(
                matches));

    std::uint16_t collision_mask = 0;


    for (int position = 0;
         position < region_length;
         ++position) {

        if (bits & (0x3u << (position * 2))) {

            collision_mask |=
                static_cast<std::uint16_t>(
                    1u << position);
        }
    }


    return collision_mask;
}


struct CurveSegment {
    std::uint8_t start;
    std::uint8_t point_count;
};


CurveSegment find_longest_clear_segment(
    std::uint16_t collision_mask)
{
    int current_start = 0;
    int current_length = 0;

    int longest_start = 0;
    int longest_length = 0;

    for (int position = 0;
         position < region_length;
         ++position) {

        const bool collides =
            (collision_mask &
             (1u << position)) != 0;

        if (collides) {
            current_length = 0;
            continue;
        }

        if (current_length == 0) {
            current_start = position;
        }

        ++current_length;

        /*
         * Strict > means equal-length ties keep
         * the earlier segment.
         */
        if (current_length > longest_length) {
            longest_length = current_length;
            longest_start = current_start;
        }
    }

    return {
        static_cast<std::uint8_t>(
            longest_start),

        static_cast<std::uint8_t>(
            longest_length)
    };
}


void find_collision_candidates(
    const RegionLocationsChunk& chunk,
    const std::vector<std::uint8_t>& curve_min_x,
    const std::vector<std::uint8_t>& curve_max_x,
    const std::vector<std::uint8_t>& curve_min_y,
    const std::vector<std::uint8_t>& curve_max_y,
    const std::vector<std::array<std::uint16_t, 16>>& temporary_curves,
    const std::vector<std::uint16_t>& curve_region_indices,
    const std::vector<std::uint16_t>& region_curve_indices,
    std::vector<std::uint8_t>& curve_start_positions,
    std::vector<std::uint8_t>& curve_point_counts)
{
    // curve_region_indices is currently not needed here because
    // iteration starts from spatial cells -> region index.
    (void)curve_region_indices;

    for (int current_cell_y = chunk_curve_min_cell; current_cell_y <= chunk_curve_max_cell; ++current_cell_y) {
        for (int current_cell_x = chunk_curve_min_cell; current_cell_x <= chunk_curve_max_cell; ++current_cell_x) {

            const int current_cell_index = current_cell_y * spatial_grid_side + current_cell_x;
            const int current_cell_count = chunk.spatial_cell_counts[current_cell_index];

            for (int current_slot = 0; current_slot < current_cell_count; ++current_slot) {

                const std::uint16_t current_region_index = chunk.spatial_cells_indices[current_cell_index][current_slot];
                const std::uint16_t current_curve_index = region_curve_indices[current_region_index];

                if (current_curve_index == UINT16_MAX) {
                    continue;
                }

                const std::uint16_t current_weight = chunk.region_weights[current_region_index];
                const CurveBounds current_bounds{
                    curve_min_x[current_curve_index],
                    curve_max_x[current_curve_index],
                    curve_min_y[current_curve_index],
                    curve_max_y[current_curve_index]
                };

                const int candidate_min_x = static_cast<int>(current_bounds.min_x) - curve_reach - curve_collision_radius;
                const int candidate_max_x = static_cast<int>(current_bounds.max_x) + curve_reach + curve_collision_radius;
                const int candidate_min_y = static_cast<int>(current_bounds.min_y) - curve_reach - curve_collision_radius;
                const int candidate_max_y = static_cast<int>(current_bounds.max_y) + curve_reach + curve_collision_radius;

                int min_cell_x = candidate_min_x / spatial_cell_side;
                int max_cell_x = candidate_max_x / spatial_cell_side;
                int min_cell_y = candidate_min_y / spatial_cell_side;
                int max_cell_y = candidate_max_y / spatial_cell_side;

                min_cell_x = std::max(min_cell_x, curve_support_min_cell);
                max_cell_x = std::min(max_cell_x, curve_support_max_cell);
                min_cell_y = std::max(min_cell_y, curve_support_min_cell);
                max_cell_y = std::min(max_cell_y, curve_support_max_cell);


                if (min_cell_x > max_cell_x || min_cell_y > max_cell_y) {
                    continue;
                }

                std::uint16_t collision_mask = 0;

                for (int cell_y = min_cell_y; cell_y <= max_cell_y; ++cell_y) {
                    for (int cell_x = min_cell_x; cell_x <= max_cell_x; ++cell_x) {

                        const int cell_index = cell_y * spatial_grid_side + cell_x;
                        const int count = chunk.spatial_cell_counts[cell_index];

                        for (int slot = 0; slot < count; ++slot) {

                            const std::uint16_t other_region_index = chunk.spatial_cells_indices[cell_index][slot];

                            if (other_region_index == current_region_index) {
                                continue;
                            }

                            // Only stronger curves can cut
                            // the current curve.
                            if (chunk.region_weights[other_region_index] < current_weight) {
                                continue;
                            }
                            else if (chunk.region_weights[other_region_index] == current_weight) {
                                if (chunk.regions_x[other_region_index] < chunk.regions_x[current_region_index]) {
                                    continue;
                                }
                                else if (chunk.regions_x[other_region_index] == chunk.regions_x[current_region_index]) {
                                    if (chunk.regions_y[other_region_index] <= chunk.regions_y[current_region_index]) {
                                        continue;
                                    }
                                }
                            }
                                


                            const std::uint16_t other_curve_index = region_curve_indices[other_region_index];

                            if (other_curve_index == UINT16_MAX) {
                                continue;
                            }

                            const CurveBounds other_bounds{
                                curve_min_x[other_curve_index],
                                curve_max_x[other_curve_index],
                                curve_min_y[other_curve_index],
                                curve_max_y[other_curve_index]
                            };

                            if (!bounds_overlap_with_margin(
                                    current_bounds,
                                    other_bounds,
                                    curve_collision_radius)) {

                                continue;
                            }

                            collision_mask |= get_collision_mask(
                                    temporary_curves[current_curve_index],
                                    temporary_curves[other_curve_index]);

                            if ((collision_mask & 0x01FFu) == 0x01FFu) {
                                break;
                            }
                        }

                        if ((collision_mask & 0x01FFu) == 0x01FFu) {
                            break;
                        }
                    }

                    if ((collision_mask & 0x01FFu) == 0x01FFu) {
                        break;
                    }
                }

                const CurveSegment segment = find_longest_clear_segment(collision_mask);

                curve_start_positions[current_curve_index] = segment.start;
                curve_point_counts[current_curve_index] = segment.point_count;
            }
        }
    }


    if (!chunk.overflow_cells.empty()) {
        /*
         * TODO:
         * Overflow points are not handled yet.
         */
    }
}


void pack_region_curves(
    RegionLocationsChunk& chunk,
    const std::vector<std::array<std::uint16_t, 16>>& temporary_curves,
    const std::vector<std::uint16_t>& curve_region_indices,
    const std::vector<std::uint16_t>& region_curve_indices,
    const std::vector<std::uint8_t>& curve_start_positions,
    const std::vector<std::uint8_t>& curve_point_counts)
{
    // Keep region_curves parallel with the region arrays.
    // Regions outside the actual 32x32 chunk remain zero.
    chunk.region_curves.assign(chunk.regions_x.size(), 0u);

    // Not needed when iteration begins from spatial cells.
    (void)curve_region_indices;


    for (int cell_y = chunk_curve_min_cell; cell_y <= chunk_curve_max_cell; ++cell_y) {

        for (int cell_x = chunk_curve_min_cell; cell_x <= chunk_curve_max_cell; ++cell_x) {

            const int cell_index = cell_y * spatial_grid_side + cell_x;
            const int count = chunk.spatial_cell_counts[cell_index];

            for (int slot = 0; slot < count; ++slot) {

                const std::uint16_t region_index = chunk.spatial_cells_indices[cell_index][slot];
                const std::uint16_t curve_index = region_curve_indices[region_index];

                if (curve_index == UINT16_MAX) {
                    continue;
                }

                const auto& curve = temporary_curves[curve_index];

                std::uint32_t packed_curve = 0u;

                for (int step = 0; step < curve_step_count; ++step) {

                    const std::uint16_t tile_a = curve[step];
                    const std::uint16_t tile_b = curve[step + 1];

                    const int x_a = static_cast<int>(tile_a >> 8);
                    const int y_a = static_cast<int>(tile_a & 0xFFu);
                    const int x_b = static_cast<int>(tile_b >> 8);
                    const int y_b = static_cast<int>(tile_b & 0xFFu);
                    const int dx = x_b - x_a;
                    const int dy = y_b - y_a;

                    const std::uint8_t direction = delta_to_curve_direction(dx, dy);

                    packed_curve |= static_cast<std::uint32_t>(direction) << (step * 3);
                }

                set_curve_start(packed_curve, curve_start_positions[curve_index]);
                set_curve_point_count(packed_curve, curve_point_counts[curve_index]);
                chunk.region_curves[region_index] = packed_curve;
            }
        }
    }
}


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

    RegionLocationsChunk chunk;

    find_region_points(
        chunk,
        seed,
        grid_x,
        grid_y);

    calculate_region_directions(
        chunk,
        seed,
        grid_x,
        grid_y);

    std::vector<std::uint8_t> curve_min_x;
    std::vector<std::uint8_t> curve_max_x;
    std::vector<std::uint8_t> curve_min_y;
    std::vector<std::uint8_t> curve_max_y;
    std::vector<std::array<std::uint16_t, 16>> temporary_curves;
    std::vector<std::uint16_t> curve_region_indices;
    std::vector<std::uint16_t> region_curve_indices(chunk.regions_x.size(), UINT16_MAX);

    create_region_curves(
        chunk,
        seed,
        grid_x,
        grid_y,
        curve_min_x,
        curve_max_x,
        curve_min_y,
        curve_max_y,
        temporary_curves,
        curve_region_indices,
        region_curve_indices);

    std::vector<std::uint8_t> curve_start_positions(temporary_curves.size(), 0);
    std::vector<std::uint8_t> curve_point_counts(temporary_curves.size(), static_cast<std::uint8_t>(region_length));

    find_collision_candidates(
        chunk,
        curve_min_x,
        curve_max_x,
        curve_min_y,
        curve_max_y,
        temporary_curves,
        curve_region_indices,
        region_curve_indices,
        curve_start_positions,
        curve_point_counts);

    pack_region_curves(
        chunk,
        temporary_curves,
        curve_region_indices,
        region_curve_indices,
        curve_start_positions,
        curve_point_counts);

    grid.emplace(
        position,
        std::move(chunk));
}