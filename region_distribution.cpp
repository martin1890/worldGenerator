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
    point_area_offset + maxima_radius; // 26

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

constexpr int chunk_cell_offset =
    chunk_side / spatial_cell_side; // 4

constexpr int curve_collision_radius = 1;

constexpr int cells_per_maxima_block =
    maxima_width / spatial_cell_side; // 2

constexpr int maxima_blocks_per_cell_row =
    spatial_grid_side /
    cells_per_maxima_block; // 5

static_assert(maxima_width == spatial_cell_side * cells_per_maxima_block);
static_assert(spatial_grid_side % cells_per_maxima_block == 0);
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
static_assert(spatial_cell_side % hash_width == 0);
static_assert(chunk_cell_offset % cells_per_maxima_block == 0);

static constexpr float markov_matrix[5][5] = {
    {0.10, 0.90, 0.00, 0.00, 0.00},
    {0.20, 0.30, 0.50, 0.00, 0.00},
    {0.00, 0.35, 0.30, 0.35, 0.00},
    {0.00, 0.00, 0.50, 0.30, 0.20},
    {0.00, 0.00, 0.00, 0.90, 0.10}
};

constexpr std::uint16_t neighbor_top_left     = 1u << 0;
constexpr std::uint16_t neighbor_top          = 1u << 1;
constexpr std::uint16_t neighbor_top_right    = 1u << 2;
constexpr std::uint16_t neighbor_left         = 1u << 3;
constexpr std::uint16_t neighbor_right        = 1u << 4;
constexpr std::uint16_t neighbor_bottom_left  = 1u << 5;
constexpr std::uint16_t neighbor_bottom       = 1u << 6;
constexpr std::uint16_t neighbor_bottom_right = 1u << 7;

std::uint16_t neighbor_bit(
    int dx,
    int dy)
{
    if (dx == -1 && dy == -1) return neighbor_top_left;
    if (dx ==  0 && dy == -1) return neighbor_top;
    if (dx ==  1 && dy == -1) return neighbor_top_right;

    if (dx == -1 && dy ==  0) return neighbor_left;
    if (dx ==  1 && dy ==  0) return neighbor_right;

    if (dx == -1 && dy ==  1) return neighbor_bottom_left;
    if (dx ==  0 && dy ==  1) return neighbor_bottom;
    return neighbor_bottom_right;
}

constexpr std::array<std::array<int, 8>, 4>
make_markov_thresholds()
{
    std::array<std::array<int, 8>, 4> thresholds{};

    for (int state = 0; state < 5; ++state) {

        float cumulative = 0.0f;

        for (int boundary = 0; boundary < 4; ++boundary) {

            cumulative += markov_matrix[state][boundary];
            const float scaled = cumulative * 256.0f;
            int threshold =static_cast<int>(scaled);

            if (static_cast<float>(threshold) < scaled) {
                ++threshold;
            }

            if (threshold < 0) {
                threshold = 0;
            }
            else if (threshold > 256) {
                threshold = 256;
            }

            /*
             * Store threshold - 1 so:
             *
             * random > threshold - 1
             *
             * is equivalent to:
             *
             * random >= threshold.
             */
            thresholds[boundary][state] = threshold - 1;
        }
    }

    /*
     * Lanes 5..7 are never selected because Markov state
     * is always in the range 0..4.
     */
    return thresholds;
}

static constexpr auto markov_thresholds = make_markov_thresholds();


inline __m256i new_random_bits_x8(__m256i x)
{
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 16));
    x = _mm256_mullo_epi32(x, _mm256_set1_epi32(static_cast<int>(0x7feb352dU)));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 15));
    x = _mm256_mullo_epi32(x, _mm256_set1_epi32(static_cast<int>(0x846ca68bU)));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 16));

    return x;
}


inline __m256i sample_next_state_x8(__m256i state, __m256i random_bits)
{
    /*
     * Use the high byte as a value in 0..255.
     */
    const __m256i random_byte = _mm256_srli_epi32(random_bits, 24);
    __m256i next_state = _mm256_setzero_si256();

    /*
     * For each of the four boundaries, fetch the cumulative
     * threshold belonging to each lane's current Markov state.
     *
     * Counting how many boundaries random has passed gives
     * the resulting state directly.
     */
    for (int boundary = 0; boundary < 4; ++boundary) {

        const __m256i threshold_table = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(markov_thresholds[boundary].data()));
        const __m256i threshold = _mm256_permutevar8x32_epi32(threshold_table, state);
        const __m256i passed = _mm256_cmpgt_epi32(random_byte, threshold);
        /*
         * passed is either 0 or -1.
         * Subtracting -1 adds one.
         */
        next_state = _mm256_sub_epi32(next_state, passed);
    }

    return next_state;
}

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


std::uint8_t delta_to_curve_direction(int dx, int dy)
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
constexpr float quarter_turn = pi * 0.5f;
constexpr float grid_angle_step = pi * 0.25f;

inline __m256i hash_u32x8(__m256i seed, __m256i x, __m256i y)
{
    const __m256i mx = _mm256_set1_epi32(static_cast<int>(374761393u));
    const __m256i my = _mm256_set1_epi32(static_cast<int>(668265263u));
    const __m256i m1 = _mm256_set1_epi32(static_cast<int>(2246822519u));
    const __m256i m2 = _mm256_set1_epi32(static_cast<int>(3266489917u));

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
    values = _mm256_and_si256(values, _mm256_set1_epi32(0xFFFF));

    const __m128i low = _mm256_castsi256_si128(values);
    const __m128i high = _mm256_extracti128_si256(values, 1);

    return _mm_packus_epi32(low, high);
}


inline __m256i greater_u16(__m256i a, __m256i b)
{
    const __m256i sign = _mm256_set1_epi16(static_cast<short>(0x8000));

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
    std::int32_t grid_y,
    int weight_min_x,
    int weight_max_x,
    int weight_min_y,
    int weight_max_y)
{
    const __m256i seed_vector = _mm256_set1_epi32(static_cast<int>(seed));
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

    for (int y = weight_min_y;
        y <= weight_max_y;
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
        for (int x = weight_min_x;
             x <= weight_max_x;
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
    int cell_index,
    std::uint8_t x,
    std::uint8_t y,
    std::uint16_t region_index)
{
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

    int first_cell_x = 0;
    int last_cell_x = spatial_grid_side - 1;

    int first_cell_y = 0;
    int last_cell_y = spatial_grid_side - 1;

    if (chunk.received_neighbor_mask & neighbor_left) {
        first_cell_x = spatial_grid_side - chunk_cell_offset; // 6
    }

    if (chunk.received_neighbor_mask &
        neighbor_right) {last_cell_x = chunk_cell_offset - 1; // 3
    }

    if (chunk.received_neighbor_mask & neighbor_top) {
        first_cell_y = spatial_grid_side - chunk_cell_offset; // 6
    }

    if (chunk.received_neighbor_mask & neighbor_bottom) {
        last_cell_y = chunk_cell_offset - 1; // 3
    }

    if (first_cell_x > last_cell_x ||
        first_cell_y > last_cell_y) {
        return;
    }
    
    const int first_block_x = first_cell_x / cells_per_maxima_block;
    const int last_block_x = last_cell_x / cells_per_maxima_block;

    const int candidate_min_x = first_cell_x * spatial_cell_side;
    const int candidate_max_x = (last_cell_x + 1) * spatial_cell_side - 1;
    const int candidate_min_y = first_cell_y * spatial_cell_side;
    const int candidate_max_y = (last_cell_y + 1) * spatial_cell_side - 1;

    const int weight_max_y = candidate_max_y + maxima_radius * 2;
    const int weight_max_x = candidate_max_x + maxima_radius * 2;
    const int weight_min_y = candidate_min_y;
    const int weight_min_x = candidate_min_x;

    WeightField weight_field;
    generate_weight_field(weight_field, 
        seed, 
        grid_x, 
        grid_y, 
        weight_min_x, 
        weight_max_x, 
        weight_min_y, 
        weight_max_y);

    chunk.regions_x.reserve(192);
    chunk.regions_y.reserve(192);
    chunk.region_weights.reserve(192);

    chunk.overflow_cells.reserve(4);


    const __m256i all_lanes =
        _mm256_set1_epi16(
            static_cast<short>(-1));

    const __m256i first_cell_lanes =
        _mm256_setr_epi64x(
            -1, -1,
             0,  0);

    const __m256i second_cell_lanes =
        _mm256_setr_epi64x(
             0,  0,
            -1, -1);

    for (int cell_y = first_cell_y; cell_y <= last_cell_y; ++cell_y) {

        const int y_begin = cell_y * spatial_cell_side;
        const int y_end = y_begin + spatial_cell_side;

        for (int block_x = first_block_x; block_x <= last_block_x; ++block_x) {
            /*
             * Each 16-wide maxima block covers exactly
             * two neighboring 8-wide spatial cells.
             */
            const int first_cell_x = block_x * cells_per_maxima_block;
            const int second_cell_x = first_cell_x + 1;
            const int first_cell_index = cell_y * spatial_grid_side + first_cell_x;
            const int second_cell_index = first_cell_index + 1;

            const bool first_cell_ready =
                chunk.cell_generation_state[
                    cell_y][first_cell_x] != 0;

            const bool second_cell_ready =
                chunk.cell_generation_state[
                    cell_y][second_cell_x] != 0;

            /*
             * Both cells already contain their complete
             * region-point data.
             */
            if (first_cell_ready &&
                second_cell_ready) {

                continue;
            }

            /*
             * Only lanes belonging to cells that still need
             * generation participate in the maxima test.
             */
            __m256i active_lanes;

            int lane_begin;
            int lane_end;

            if (first_cell_ready) {
                active_lanes = second_cell_lanes;
                lane_begin = spatial_cell_side;
                lane_end = maxima_width;
            }
            else if (second_cell_ready) {
                active_lanes = first_cell_lanes;
                lane_begin = 0;
                lane_end = spatial_cell_side;
            }
            else {
                active_lanes = all_lanes;
                lane_begin = 0;
                lane_end = maxima_width;
            }

            const int x = block_x * maxima_width;
            const int field_x = x + maxima_radius;

            /*
             * Process the eight rows belonging to this
             * pair of spatial cells.
             */
            for (int y = y_begin; y < y_end; ++y) {

                const int field_y = y + maxima_radius;

                const std::uint16_t* center_pointer =
                    weight_field.data() +
                    static_cast<std::size_t>(field_y) *
                        weight_field_stride +
                    field_x;

                const __m256i center =
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            center_pointer));

                /*
                 * Ready cells start with zero lanes, so they
                 * take no further part in the maxima test.
                 */
                __m256i mask = active_lanes;


                for (int dy = -maxima_radius; dy <= maxima_radius; ++dy) {
                    for (int dx = -maxima_radius; dx <= maxima_radius; ++dx) {

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

                        mask =
                            _mm256_and_si256(
                                mask,
                                greater_u16(
                                    center,
                                    neighbor));

                        if (_mm256_testz_si256(mask, mask)) {
                            break;
                        }
                    }

                    if (_mm256_testz_si256(mask, mask)) {
                        break;
                    }
                }

                const std::uint32_t bits =
                    static_cast<std::uint32_t>(
                        _mm256_movemask_epi8(
                            mask));

                if (bits == 0) {
                    continue;
                }

                /*
                 * Extract only lanes belonging to cells that
                 * were not already available.
                 */
                for (int lane = lane_begin; lane < lane_end; ++lane) {

                    const std::uint32_t lane_mask = 0x3u << (lane * 2);

                    if ((bits & lane_mask) != lane_mask) {
                        continue;
                    }

                    const std::uint8_t point_x =
                        static_cast<std::uint8_t>(
                            x + lane);

                    const std::uint8_t point_y =
                        static_cast<std::uint8_t>(
                            y);

                    const std::uint16_t region_index =
                        static_cast<std::uint16_t>(
                            chunk.regions_x.size());

                    chunk.regions_x.push_back(point_x);
                    chunk.regions_y.push_back(point_y);
                    chunk.region_weights.push_back(center_pointer[lane]);
                    chunk.region_directions.push_back(0.0f);
                    chunk.region_curve_indices.push_back(UINT16_MAX);

                    /*
                     * Lanes 0..7 belong to the first cell.
                     * Lanes 8..15 belong to the second.
                     */
                    const int cell_index =
                        lane < spatial_cell_side
                            ? first_cell_index
                            : second_cell_index;


                    add_to_spatial_cell(
                        chunk,
                        cell_index,
                        point_x,
                        point_y,
                        region_index);
                }
            }

            /*
             * Both cells are now complete.
             *
             * A previously ready cell remains ready, while
             * a generated cell becomes ready here.
             */

            if (!first_cell_ready) {
                chunk.cell_generation_state[
                    cell_y][first_cell_x] = 1;
            }
            
            if (!second_cell_ready) {
                chunk.cell_generation_state[
                    cell_y][second_cell_x] = 1;
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
            
            if (chunk.cell_generation_state[cell_y][cell_x] >= 2) {
                continue;
            }

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
        
        if (chunk.cell_generation_state[cell_y][cell_x] < 2) {
            chunk.cell_generation_state[cell_y][cell_x] = 2;
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
    std::int32_t grid_y)
{
    const std::size_t point_count =
        chunk.regions_x.size();

    /*
     * Existing curves may have been received from neighboring chunks.
     * This list contains only regions whose curves must be generated now.
     */
    std::vector<std::uint16_t> curve_region_indices;
    curve_region_indices.reserve(point_count);

    /*
     * New curves are appended after curves already stored in the chunk.
     */
    const std::size_t first_new_curve_index =
        chunk.temporary_curves.size();


    for (int cell_y = curve_support_min_cell;
         cell_y <= curve_support_max_cell;
         ++cell_y) {

        for (int cell_x = curve_support_min_cell;
             cell_x <= curve_support_max_cell;
             ++cell_x) {

            if (chunk.cell_generation_state[cell_y][cell_x] >= 3) {
                continue;
            }

            const int cell_index =
                cell_y * spatial_grid_side +
                cell_x;

            const int count =
                chunk.spatial_cell_counts[cell_index];


            for (int slot = 0;
                 slot < count;
                 ++slot) {

                const std::uint16_t point_index =
                    chunk.spatial_cells_indices[
                        cell_index][slot];

                /*
                 * Normally every region in an unfinished curve cell
                 * has UINT16_MAX here. This also makes the function
                 * safe if a region was already supplied externally.
                 */
                if (chunk.region_curve_indices[point_index] !=
                    UINT16_MAX) {

                    continue;
                }

                const std::size_t curve_index =
                    first_new_curve_index +
                    curve_region_indices.size();

                chunk.region_curve_indices[point_index] =
                    static_cast<std::uint16_t>(
                        curve_index);

                curve_region_indices.push_back(
                    point_index);
            }
        }
    }


    const std::size_t new_curve_count =
        curve_region_indices.size();

    const std::size_t total_curve_count =
        first_new_curve_index +
        new_curve_count;


    chunk.curve_min_x.resize(total_curve_count);
    chunk.curve_max_x.resize(total_curve_count);
    chunk.curve_min_y.resize(total_curve_count);
    chunk.curve_max_y.resize(total_curve_count);

    chunk.temporary_curves.resize(
        total_curve_count);

    if (new_curve_count == 0) {

        for (int cell_y = curve_support_min_cell;
            cell_y <= curve_support_max_cell;
            ++cell_y) {

            for (int cell_x = curve_support_min_cell;
                cell_x <= curve_support_max_cell;
                ++cell_x) {

                chunk.cell_generation_state[
                    cell_y][cell_x] = 3;
            }
        }

        return;
    }


    /*
     * Integer direction lookup tables.
     *
     * _mm256_permutevar8x32_epi32 lets each SIMD lane select
     * its own direction without a gather.
     */
    const __m256i dx_lookup =
        _mm256_setr_epi32(
             1,  1,  0, -1,
            -1, -1,  0,  1);

    const __m256i dy_lookup =
        _mm256_setr_epi32(
             0,  1,  1,  1,
             0, -1, -1, -1);


    const __m256i direction_mask =
        _mm256_set1_epi32(7);

    const __m256i one =
        _mm256_set1_epi32(1);

    const __m256i state_center =
        _mm256_set1_epi32(2);


    /*
     * main_angle / grid_angle_step
     *
     * Multiplication is cheaper than performing eight divisions.
     */
    const __m256 inverse_grid_angle_step =
        _mm256_set1_ps(
            1.0f / grid_angle_step);

    /*
     * Direction random uses the high 24 random bits.
     *
     * 2^24 fits comfortably in signed int32 and matches the
     * useful precision of a float.
     */
    constexpr float direction_random_scale =
        16777216.0f;

    const __m256 direction_random_scale_vector =
        _mm256_set1_ps(
            direction_random_scale);


    /*
     * World coordinate base.
     *
     * Unsigned wrap is intentional and matches the deterministic
     * coordinate hashing used elsewhere.
     */
    const std::uint32_t world_base_x =
        static_cast<std::uint32_t>(grid_x) *
            static_cast<std::uint32_t>(chunk_side) -
        static_cast<std::uint32_t>(point_area_offset);

    const std::uint32_t world_base_y =
        static_cast<std::uint32_t>(grid_y) *
            static_cast<std::uint32_t>(chunk_side) -
        static_cast<std::uint32_t>(point_area_offset);

    const __m256i world_base_x_vector =
        _mm256_set1_epi32(
            static_cast<int>(world_base_x));

    const __m256i world_base_y_vector =
        _mm256_set1_epi32(
            static_cast<int>(world_base_y));

    const __m256i random_seed =
        _mm256_set1_epi32(
            static_cast<int>(
                seed ^ 0x6C8E9CF5u));


    /*
     * Temporary AoSoA working storage.
     *
     * Each row is one curve position.
     * Each column is one SIMD lane / curve.
     *
     * After generation it is transposed into the existing AoS
     * chunk.temporary_curves layout used by collision detection.
     */
    alignas(32) std::uint32_t
        generated_points[region_length][8];

    alignas(32) int point_indices[8];
    alignas(32) int point_x_values[8];
    alignas(32) int point_y_values[8];

    alignas(32) int min_x_values[8];
    alignas(32) int max_x_values[8];
    alignas(32) int min_y_values[8];
    alignas(32) int max_y_values[8];


    for (std::size_t batch_start = 0; batch_start < new_curve_count; batch_start += 8) 
    {
        const int valid_count =
            static_cast<int>(
                std::min<std::size_t>(
                    8,
                    new_curve_count - batch_start));

        /*
         * AVX2 has no need for a separate scalar tail here.
         *
         * Invalid lanes duplicate the last real curve in the batch.
         * They execute normally, but their results are simply not
         * written to the output.
         */
        const std::uint16_t fallback_region =
            curve_region_indices[
                batch_start +
                static_cast<std::size_t>(
                    valid_count - 1)];

        for (int lane = 0;
             lane < 8;
             ++lane) {

            const std::uint16_t point_index =
                lane < valid_count
                    ? curve_region_indices[
                        batch_start +
                        static_cast<std::size_t>(lane)]
                    : fallback_region;

            point_indices[lane] =
                static_cast<int>(point_index);

            point_x_values[lane] =
                static_cast<int>(
                    chunk.regions_x[point_index]);

            point_y_values[lane] =
                static_cast<int>(
                    chunk.regions_y[point_index]);
        }


        const __m256i region_indices =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    point_indices));

        __m256i curve_x =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    point_x_values));

        __m256i curve_y =
            _mm256_load_si256(
                reinterpret_cast<const __m256i*>(
                    point_y_values));


        __m256i min_x = curve_x;
        __m256i max_x = curve_x;
        __m256i min_y = curve_y;
        __m256i max_y = curve_y;


        /*
         * Gather eight main angles.
         */
        const __m256 main_angle =
            _mm256_i32gather_ps(
                chunk.region_directions.data(),
                region_indices,
                sizeof(float));


        /*
         * Quantize main_angle ONCE for the whole curve.
         *
         * grid_position = integer direction + fractional position.
         *
         * Adding a Markov state later only changes the integer part,
         * because every state offset is exactly one 45-degree step.
         */
        const __m256 grid_position =
            _mm256_mul_ps(
                main_angle,
                inverse_grid_angle_step);

        const __m256 base_direction_float =
            _mm256_floor_ps(
                grid_position);

        const __m256 fraction =
            _mm256_sub_ps(
                grid_position,
                base_direction_float);

        const __m256i base_direction =
            _mm256_cvttps_epi32(
                base_direction_float);


        /*
         * Convert the fractional direction probability once into an
         * integer threshold in 0..2^24.
         */
        const __m256i direction_threshold =
            _mm256_cvttps_epi32(
                _mm256_mul_ps(
                    fraction,
                    direction_random_scale_vector));


        /*
         * Initial independent PRNG state for all eight curves.
         */
        const __m256i world_x =
            _mm256_add_epi32(
                world_base_x_vector,
                curve_x);

        const __m256i world_y =
            _mm256_add_epi32(
                world_base_y_vector,
                curve_y);

        __m256i random_bits =
            hash_u32x8(
                random_seed,
                world_x,
                world_y);


        /*
         * Every curve starts in Markov state 2: straight.
         */
        __m256i state =
            state_center;


        /*
         * Store position zero.
         */
        const __m256i initial_tiles =
            _mm256_or_si256(
                _mm256_slli_epi32(
                    curve_x,
                    8),
                curve_y);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(
                generated_points[0]),
            initial_tiles);


        for (int step = 0;
             step < curve_step_count;
             ++step) {

            /*
             * First random value chooses between the two neighboring
             * grid directions.
             */
            random_bits =
                new_random_bits_x8(
                    random_bits);

            /*
             * The high 24 bits form an integer random value
             * in 0..2^24-1.
             */
            const __m256i direction_random =
                _mm256_srli_epi32(
                    random_bits,
                    8);

            /*
             * random < threshold selects the upper grid direction.
             */
            const __m256i use_upper_mask =
                _mm256_cmpgt_epi32(
                    direction_threshold,
                    direction_random);

            const __m256i use_upper =
                _mm256_and_si256(
                    use_upper_mask,
                    one);


            /*
             * Markov state:
             *
             * 0 -> -2 grid directions
             * 1 -> -1
             * 2 ->  0
             * 3 -> +1
             * 4 -> +2
             *
             * No +-90 degree pruning is needed anymore.
             */
            __m256i grid_direction =
                _mm256_add_epi32(
                    base_direction,
                    _mm256_sub_epi32(
                        state,
                        state_center));

            grid_direction =
                _mm256_add_epi32(
                    grid_direction,
                    use_upper);

            grid_direction =
                _mm256_and_si256(
                    grid_direction,
                    direction_mask);


            /*
             * Eight independent direction lookups without gathers.
             */
            const __m256i dx =
                _mm256_permutevar8x32_epi32(
                    dx_lookup,
                    grid_direction);

            const __m256i dy =
                _mm256_permutevar8x32_epi32(
                    dy_lookup,
                    grid_direction);


            curve_x =
                _mm256_add_epi32(
                    curve_x,
                    dx);

            curve_y =
                _mm256_add_epi32(
                    curve_y,
                    dy);


            min_x =
                _mm256_min_epi32(
                    min_x,
                    curve_x);

            max_x =
                _mm256_max_epi32(
                    max_x,
                    curve_x);

            min_y =
                _mm256_min_epi32(
                    min_y,
                    curve_y);

            max_y =
                _mm256_max_epi32(
                    max_y,
                    curve_y);


            const __m256i tiles =
                _mm256_or_si256(
                    _mm256_slli_epi32(
                        curve_x,
                        8),
                    curve_y);

            _mm256_store_si256(
                reinterpret_cast<__m256i*>(
                    generated_points[step + 1]),
                tiles);


            /*
             * Second random value advances the Markov chain.
             */
            random_bits =
                new_random_bits_x8(
                    random_bits);

            state =
                sample_next_state_x8(
                    state,
                    random_bits);
        }


        _mm256_store_si256(
            reinterpret_cast<__m256i*>(
                min_x_values),
            min_x);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(
                max_x_values),
            max_x);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(
                min_y_values),
            min_y);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(
                max_y_values),
            max_y);


        /*
         * Transpose the temporary AoSoA batch into the existing
         * curve-major AoS representation.
         *
         * Only real lanes are stored, so the padded SIMD tail costs
         * no special scalar curve-generation path.
         */
        for (int lane = 0;
             lane < valid_count;
             ++lane) {

            const std::size_t curve_index =
                first_new_curve_index +
                batch_start +
                static_cast<std::size_t>(lane);

            auto& temporary_curve =
                chunk.temporary_curves[curve_index];

            temporary_curve.fill(0xFFFFu);

            for (int position = 0;
                 position < region_length;
                 ++position) {

                temporary_curve[position] =
                    static_cast<std::uint16_t>(
                        generated_points[position][lane]);
            }


            chunk.curve_min_x[curve_index] =
                static_cast<std::uint8_t>(
                    min_x_values[lane]);

            chunk.curve_max_x[curve_index] =
                static_cast<std::uint8_t>(
                    max_x_values[lane]);

            chunk.curve_min_y[curve_index] =
                static_cast<std::uint8_t>(
                    min_y_values[lane]);

            chunk.curve_max_y[curve_index] =
                static_cast<std::uint8_t>(
                    max_y_values[lane]);
        }
    }

    for (int cell_y = curve_support_min_cell;
        cell_y <= curve_support_max_cell;
        ++cell_y) {

        for (int cell_x = curve_support_min_cell;
            cell_x <= curve_support_max_cell;
            ++cell_x) {

            chunk.cell_generation_state[
                cell_y][cell_x] = 3;
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
    std::vector<std::uint8_t>& curve_start_positions,
    std::vector<std::uint8_t>& curve_point_counts)
{

    for (int current_cell_y = chunk_curve_min_cell; current_cell_y <= chunk_curve_max_cell; ++current_cell_y) {
        for (int current_cell_x = chunk_curve_min_cell; current_cell_x <= chunk_curve_max_cell; ++current_cell_x) {

            const int current_cell_index = current_cell_y * spatial_grid_side + current_cell_x;
            const int current_cell_count = chunk.spatial_cell_counts[current_cell_index];

            for (int current_slot = 0; current_slot < current_cell_count; ++current_slot) {

                const std::uint16_t current_region_index = chunk.spatial_cells_indices[current_cell_index][current_slot];
                const std::uint16_t current_curve_index = chunk.region_curve_indices[current_region_index];

                if (current_curve_index == UINT16_MAX) {
                    continue;
                }

                const std::uint16_t current_weight = chunk.region_weights[current_region_index];
                const CurveBounds current_bounds{
                    chunk.curve_min_x[current_curve_index],
                    chunk.curve_max_x[current_curve_index],
                    chunk.curve_min_y[current_curve_index],
                    chunk.curve_max_y[current_curve_index]
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
                                


                            const std::uint16_t other_curve_index = chunk.region_curve_indices[other_region_index];

                            if (other_curve_index == UINT16_MAX) {
                                continue;
                            }

                            const CurveBounds other_bounds{
                                chunk.curve_min_x[other_curve_index],
                                chunk.curve_max_x[other_curve_index],
                                chunk.curve_min_y[other_curve_index],
                                chunk.curve_max_y[other_curve_index]
                            };

                            if (!bounds_overlap_with_margin(
                                    current_bounds,
                                    other_bounds,
                                    curve_collision_radius)) {

                                continue;
                            }

                            collision_mask |= get_collision_mask(
                                    chunk.temporary_curves[current_curve_index],
                                    chunk.temporary_curves[other_curve_index]);

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
    const std::vector<std::uint8_t>& curve_start_positions,
    const std::vector<std::uint8_t>& curve_point_counts)
{
    // Keep region_curves parallel with the region arrays.
    // Regions outside the actual 32x32 chunk remain zero.
    chunk.region_curves.assign(chunk.regions_x.size(), 0u);


    for (int cell_y = chunk_curve_min_cell; cell_y <= chunk_curve_max_cell; ++cell_y) {

        for (int cell_x = chunk_curve_min_cell; cell_x <= chunk_curve_max_cell; ++cell_x) {

            const int cell_index = cell_y * spatial_grid_side + cell_x;
            const int count = chunk.spatial_cell_counts[cell_index];

            for (int slot = 0; slot < count; ++slot) {

                const std::uint16_t region_index = chunk.spatial_cells_indices[cell_index][slot];
                const std::uint16_t curve_index = chunk.region_curve_indices[region_index];

                if (curve_index == UINT16_MAX) {
                    continue;
                }

                const auto& curve = chunk.temporary_curves[curve_index];

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

std::uint16_t find_region_in_cell(
    const RegionLocationsChunk& chunk,
    int cell_index,
    std::uint8_t x,
    std::uint8_t y)
{
    const int count =
        chunk.spatial_cell_counts[cell_index];

    for (int slot = 0; slot < count; ++slot) {

        const std::uint16_t region_index =
            chunk.spatial_cells_indices[
                cell_index][slot];

        if (chunk.regions_x[region_index] == x &&
            chunk.regions_y[region_index] == y) {

            return region_index;
        }
    }

    return UINT16_MAX;
}

void copy_cells_to_neighbor(
    const RegionLocationsChunk& source,
    RegionLocationsChunk& target,
    int dx,
    int dy)
{
    bool point_overlap_complete = true;

    const int source_max_y = std::min(spatial_grid_side - 1, spatial_grid_side - 1 + dy * chunk_cell_offset);
    const int source_max_x = std::min(spatial_grid_side - 1, spatial_grid_side - 1 + dx * chunk_cell_offset);    
    const int source_min_x = std::max(0, dx * chunk_cell_offset);
    const int source_min_y = std::max(0, dy * chunk_cell_offset);    

    const int coordinate_offset_x = -dx * chunk_side;
    const int coordinate_offset_y = -dy * chunk_side;

    const std::size_t expected_regions =
        target.regions_x.size() +
        source.regions_x.size();

    target.regions_x.reserve(expected_regions);
    target.regions_y.reserve(expected_regions);
    target.region_weights.reserve(expected_regions);
    target.region_directions.reserve(expected_regions);
    target.region_curve_indices.reserve(expected_regions);

    const std::size_t expected_curves =
        target.temporary_curves.size() +
        source.temporary_curves.size();

    target.temporary_curves.reserve(expected_curves);

    target.curve_min_x.reserve(expected_curves);
    target.curve_max_x.reserve(expected_curves);
    target.curve_min_y.reserve(expected_curves);
    target.curve_max_y.reserve(expected_curves);

    const __m256i curve_x_offset =_mm256_set1_epi16(static_cast<short>(coordinate_offset_x * 256));
    const __m256i curve_y_offset = _mm256_set1_epi16(static_cast<short>(coordinate_offset_y));
    const __m256i curve_x_mask = _mm256_set1_epi16(static_cast<short>(0xFF00));
    const __m256i curve_y_mask = _mm256_set1_epi16(static_cast<short>(0x00FF));
    const __m256i valid_curve_positions =
        _mm256_setr_epi16(
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            -1,
            0,  0,  0,
            0,  0,  0,  0);

    /*
     * Cell generation state:
     *
     * 0 = nothing
     * 1 = points
     * 2 = points + directions
     * 3 = points + directions + curves
     */
    for (int source_y = source_min_y; source_y <= source_max_y; ++source_y) {

        const int target_y = source_y - dy * chunk_cell_offset;

        for (int source_x = source_min_x; source_x <= source_max_x; ++source_x) {

            const int target_x = source_x - dx * chunk_cell_offset;
            const int source_cell_index = source_y * spatial_grid_side + source_x;
            const int target_cell_index = target_y * spatial_grid_side + target_x;
            const std::uint8_t source_state = source.cell_generation_state[source_y][source_x];
            std::uint8_t& target_state = target.cell_generation_state[target_y][target_x];
            /*
             * Directions and curves are only useful in support cells
             * 1..8. Cells on the outer edge only need points.
             */
            const bool target_support_cell =
                target_x >= curve_support_min_cell &&
                target_x <= curve_support_max_cell &&
                target_y >= curve_support_min_cell &&
                target_y <= curve_support_max_cell;

            const std::uint8_t available_state =
                target_support_cell
                    ? source_state
                    : std::min<std::uint8_t>(source_state, 1);

            /*
            * This neighbor overlap is only complete if the source
            * has complete point data for every overlapping cell.
            */
            if (source_state < 1) {
                point_overlap_complete = false;
            }

            if (target_state >= available_state) {
                continue;
            }
 
            const std::uint8_t old_target_state =
                target_state;

            const int count =
                source.spatial_cell_counts[source_cell_index];

            for (int slot = 0; slot < count; ++slot) {

                const std::uint16_t source_region_index =
                    source.spatial_cells_indices[
                        source_cell_index][slot];

                const std::uint8_t transformed_x =
                    static_cast<std::uint8_t>(
                        static_cast<int>(
                            source.regions_x[source_region_index]) +
                        coordinate_offset_x);

                const std::uint8_t transformed_y =
                    static_cast<std::uint8_t>(
                        static_cast<int>(
                            source.regions_y[source_region_index]) +
                        coordinate_offset_y);


                std::uint16_t target_region_index;


                /*
                * If the target cell had no points, append them directly.
                * Otherwise the same world-space region already exists and
                * only needs to be upgraded.
                */
                if (old_target_state == 0) {

                    target_region_index =
                        static_cast<std::uint16_t>(
                            target.regions_x.size());

                    target.regions_x.push_back(
                        transformed_x);

                    target.regions_y.push_back(
                        transformed_y);

                    target.region_weights.push_back(
                        source.region_weights[
                            source_region_index]);

                    target.region_directions.push_back(
                        available_state >= 2
                            ? source.region_directions[
                                source_region_index]
                            : 0.0f);

                    target.region_curve_indices.push_back(
                        UINT16_MAX);


                    add_to_spatial_cell(
                        target,
                        target_cell_index,
                        transformed_x,
                        transformed_y,
                        target_region_index);
                }
                else {

                    target_region_index =
                        find_region_in_cell(
                            target,
                            target_cell_index,
                            transformed_x,
                            transformed_y);

                    if (target_region_index == UINT16_MAX) {
                        continue;
                    }


                    if (available_state >= 2 &&
                        old_target_state < 2) {

                        target.region_directions[
                            target_region_index] =
                            source.region_directions[
                                source_region_index];
                    }
                }


                if (available_state < 3 ||
                    old_target_state >= 3) {

                    continue;
                }


                const std::uint16_t source_curve_index =
                    source.region_curve_indices[
                        source_region_index];

                if (source_curve_index == UINT16_MAX) {
                    continue;
                }


                const std::uint16_t target_curve_index =
                    static_cast<std::uint16_t>(
                        target.temporary_curves.size());


                std::array<std::uint16_t, 16>
                    transformed_curve;


                const auto& source_curve =
                    source.temporary_curves[
                        source_curve_index];

            const __m256i source_tiles =
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        source_curve.data()));

            const __m256i transformed_curve_x =
                _mm256_add_epi16(
                    _mm256_and_si256(
                        source_tiles,
                        curve_x_mask),
                    curve_x_offset);

            const __m256i transformed_curve_y =
                _mm256_add_epi16(
                    _mm256_and_si256(
                        source_tiles,
                        curve_y_mask),
                    curve_y_offset);

            const __m256i transformed_tiles =
                _mm256_or_si256(
                    transformed_curve_x,
                    transformed_curve_y);

            /*
            * Preserve the unused 0xFFFF tail.
            */
            const __m256i final_tiles =
                _mm256_blendv_epi8(
                    source_tiles,
                    transformed_tiles,
                    valid_curve_positions);

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(
                    transformed_curve.data()),
                final_tiles);


                target.temporary_curves.push_back(
                    transformed_curve);

                target.curve_min_x.push_back(
                    static_cast<std::uint8_t>(
                        static_cast<int>(
                            source.curve_min_x[
                                source_curve_index]) +
                        coordinate_offset_x));

                target.curve_max_x.push_back(
                    static_cast<std::uint8_t>(
                        static_cast<int>(
                            source.curve_max_x[
                                source_curve_index]) +
                        coordinate_offset_x));

                target.curve_min_y.push_back(
                    static_cast<std::uint8_t>(
                        static_cast<int>(
                            source.curve_min_y[
                                source_curve_index]) +
                        coordinate_offset_y));

                target.curve_max_y.push_back(
                    static_cast<std::uint8_t>(
                        static_cast<int>(
                            source.curve_max_y[
                                source_curve_index]) +
                        coordinate_offset_y));

                target.region_curve_indices[
                    target_region_index] =
                    target_curve_index;
            }


            target_state =
                available_state;
        }
    }
    if (point_overlap_complete) {
        target.received_neighbor_mask |=
            neighbor_bit(
                -dx,
                -dy);
    }
}

void adjacent_chunks(
    std::int32_t grid_x,
    std::int32_t grid_y,
    std::unordered_map<
        std::pair<int, int>,
        RegionLocationsChunk,
        PairHash>& grid)
{
    const std::pair<int, int> source_position{
        grid_x,
        grid_y
    };

    const RegionLocationsChunk& source =
        grid.at(source_position);


    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {

            if (dx == 0 && dy == 0) {
                continue;
            }


            const std::pair<int, int> target_position{
                grid_x + dx,
                grid_y + dy
            };


            auto target_it =
                grid.try_emplace(
                    target_position,
                    RegionLocationsChunk{}).first;

            RegionLocationsChunk& target =
                target_it->second;


            if (target.generation_state ==
                RegionLocationsChunk::
                    ChunkGenerationState::Finished) {

                continue;
            }


            copy_cells_to_neighbor(
                source,
                target,
                dx,
                dy);
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

    auto it = grid.try_emplace(position, RegionLocationsChunk{}).first;
    RegionLocationsChunk& chunk = it->second;

    if (chunk.generation_state ==
        RegionLocationsChunk::ChunkGenerationState::Finished) {

        return;
    }

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

    create_region_curves(
        chunk,
        seed,
        grid_x,
        grid_y);

    std::vector<std::uint8_t> curve_start_positions(chunk.temporary_curves.size(), 0);
    std::vector<std::uint8_t> curve_point_counts(chunk.temporary_curves.size(), static_cast<std::uint8_t>(region_length));

    adjacent_chunks (
        grid_x,
        grid_y,
        grid
    );

    find_collision_candidates(
        chunk,
        curve_start_positions,
        curve_point_counts);

    pack_region_curves(
        chunk,
        curve_start_positions,
        curve_point_counts);

    chunk.generation_state =
    RegionLocationsChunk::ChunkGenerationState::Finished;
}