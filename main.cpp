#include <MiniFB.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "region_distribution.hpp"


constexpr int width = 800;
constexpr int height = 600;

constexpr std::uint32_t world_seed = 12345u;

constexpr float player_speed = 0.18f;
constexpr int pixels_per_tile = 12;

constexpr int curve_line_radius = 2;
constexpr int curve_point_radius = 2;
constexpr int origin_point_radius = 3;
constexpr int player_point_radius = 4;

constexpr float direction_arrow_length_tiles = 1.8f;

constexpr std::uint32_t black = 0x00000000u;
constexpr std::uint32_t red = 0x00FF0000u;
constexpr std::uint32_t white = 0x00FFFFFFu;


using RegionGrid = std::unordered_map<
    std::pair<int, int>,
    RegionLocationsChunk,
    PairHash>;


struct DecodedCurvePoints {
    std::array<int, region_length> x{};
    std::array<int, region_length> y{};
};


void draw_square(
    std::vector<std::uint32_t>& buffer,
    int center_x,
    int center_y,
    int radius,
    std::uint32_t color)
{
    const int min_x = std::max(0, center_x - radius);
    const int max_x = std::min(width - 1, center_x + radius);

    const int min_y = std::max(0, center_y - radius);
    const int max_y = std::min(height - 1, center_y + radius);

    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            buffer[y * width + x] = color;
        }
    }
}


void draw_thick_line(
    std::vector<std::uint32_t>& buffer,
    int x0,
    int y0,
    int x1,
    int y1,
    int radius,
    std::uint32_t color)
{
    const int dx = x1 - x0;
    const int dy = y1 - y0;

    const int steps = std::max(
        std::abs(dx),
        std::abs(dy));

    if (steps == 0) {
        draw_square(
            buffer,
            x0,
            y0,
            radius,
            color);

        return;
    }

    for (int i = 0; i <= steps; ++i) {

        const float t =
            static_cast<float>(i) /
            static_cast<float>(steps);

        const int x =
            static_cast<int>(
                std::lround(
                    static_cast<float>(x0) +
                    t * static_cast<float>(dx)));

        const int y =
            static_cast<int>(
                std::lround(
                    static_cast<float>(y0) +
                    t * static_cast<float>(dy)));

        draw_square(
            buffer,
            x,
            y,
            radius,
            color);
    }
}


int world_to_chunk(
    float world_coordinate)
{
    return static_cast<int>(
        std::floor(
            world_coordinate /
            static_cast<float>(chunk_side)));
}


std::uint32_t hash_color(
    std::uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;

    value ^= value >> 15;
    value *= 0x846ca68bu;

    value ^= value >> 16;

    return value;
}


std::uint32_t curve_color(
    int chunk_x,
    int chunk_y,
    std::size_t region_index)
{
    std::uint32_t h =
        static_cast<std::uint32_t>(chunk_x) *
            0x9E3779B9u ^
        static_cast<std::uint32_t>(chunk_y) *
            0x85EBCA6Bu ^
        static_cast<std::uint32_t>(region_index) *
            0xC2B2AE35u ^
        world_seed;

    h = hash_color(h);

    const std::uint8_t r =
        static_cast<std::uint8_t>(
            80u + (h & 0x7Fu));

    const std::uint8_t g =
        static_cast<std::uint8_t>(
            80u + ((h >> 8) & 0x7Fu));

    const std::uint8_t b =
        static_cast<std::uint8_t>(
            80u + ((h >> 16) & 0x7Fu));

    return
        (static_cast<std::uint32_t>(r) << 16) |
        (static_cast<std::uint32_t>(g) << 8) |
        static_cast<std::uint32_t>(b);
}


DecodedCurvePoints decode_curve_points(
    int world_start_x,
    int world_start_y,
    std::uint32_t curve)
{
    DecodedCurvePoints points{};

    points.x[0] = world_start_x;
    points.y[0] = world_start_y;

    int current_x = world_start_x;
    int current_y = world_start_y;

    for (int step = 0;
         step < curve_step_count;
         ++step) {

        const std::uint8_t direction =
            get_curve_step(
                curve,
                step);

        current_x +=
            curve_dx[direction];

        current_y +=
            curve_dy[direction];

        points.x[step + 1] =
            current_x;

        points.y[step + 1] =
            current_y;
    }

    return points;
}


void draw_direction_arrow(
    std::vector<std::uint32_t>& buffer,
    float origin_world_x,
    float origin_world_y,
    float direction_angle,
    float player_x,
    float player_y,
    std::uint32_t color)
{
    const float end_world_x =
        origin_world_x +
        std::cos(direction_angle) *
        direction_arrow_length_tiles;

    const float end_world_y =
        origin_world_y +
        std::sin(direction_angle) *
        direction_arrow_length_tiles;


    const int screen_x0 =
        width / 2 +
        static_cast<int>(
            std::lround(
                (origin_world_x - player_x) *
                pixels_per_tile));

    const int screen_y0 =
        height / 2 +
        static_cast<int>(
            std::lround(
                (origin_world_y - player_y) *
                pixels_per_tile));


    const int screen_x1 =
        width / 2 +
        static_cast<int>(
            std::lround(
                (end_world_x - player_x) *
                pixels_per_tile));

    const int screen_y1 =
        height / 2 +
        static_cast<int>(
            std::lround(
                (end_world_y - player_y) *
                pixels_per_tile));


    draw_thick_line(
        buffer,
        screen_x0,
        screen_y0,
        screen_x1,
        screen_y1,
        1,
        color);
}


void draw_chunk_boundaries(
    std::vector<std::uint32_t>& buffer,
    float player_x,
    float player_y)
{
    const float half_view_width_tiles =
        static_cast<float>(width) /
        (2.0f * pixels_per_tile);

    const float half_view_height_tiles =
        static_cast<float>(height) /
        (2.0f * pixels_per_tile);


    const float min_world_x =
        player_x -
        half_view_width_tiles;

    const float max_world_x =
        player_x +
        half_view_width_tiles;

    const float min_world_y =
        player_y -
        half_view_height_tiles;

    const float max_world_y =
        player_y +
        half_view_height_tiles;


    const int first_chunk_x =
        world_to_chunk(min_world_x);

    const int last_chunk_x =
        world_to_chunk(max_world_x);

    const int first_chunk_y =
        world_to_chunk(min_world_y);

    const int last_chunk_y =
        world_to_chunk(max_world_y);


    // Vertical chunk boundaries.
    for (int chunk_x = first_chunk_x;
         chunk_x <= last_chunk_x + 1;
         ++chunk_x) {

        const float world_x =
            static_cast<float>(
                chunk_x * chunk_side);

        const int screen_x =
            width / 2 +
            static_cast<int>(
                std::lround(
                    (world_x - player_x) *
                    pixels_per_tile));

        if (screen_x < 0 ||
            screen_x >= width) {

            continue;
        }

        for (int screen_y = 0;
             screen_y < height;
             ++screen_y) {

            buffer[
                screen_y * width +
                screen_x] = red;
        }
    }


    // Horizontal chunk boundaries.
    for (int chunk_y = first_chunk_y;
         chunk_y <= last_chunk_y + 1;
         ++chunk_y) {

        const float world_y =
            static_cast<float>(
                chunk_y * chunk_side);

        const int screen_y =
            height / 2 +
            static_cast<int>(
                std::lround(
                    (world_y - player_y) *
                    pixels_per_tile));

        if (screen_y < 0 ||
            screen_y >= height) {

            continue;
        }

        for (int screen_x = 0;
             screen_x < width;
             ++screen_x) {

            buffer[
                screen_y * width +
                screen_x] = red;
        }
    }
}


int main()
{
    std::vector<std::uint32_t> buffer(
        width * height,
        black);

    mfb_window* window =
        mfb_open(
            "Curve viewer",
            width,
            height);

    if (window == nullptr) {
        return 1;
    }


    RegionGrid grid;

    float player_x = 0.0f;
    float player_y = 0.0f;

    mfb_set_target_fps(60);


    while (mfb_wait_sync(window)) {

        const std::uint8_t* keys =
            mfb_get_key_buffer(window);

        float movement_x = 0.0f;
        float movement_y = 0.0f;


        if (keys[KB_KEY_W]) {
            movement_y -= 1.0f;
        }

        if (keys[KB_KEY_S]) {
            movement_y += 1.0f;
        }

        if (keys[KB_KEY_A]) {
            movement_x -= 1.0f;
        }

        if (keys[KB_KEY_D]) {
            movement_x += 1.0f;
        }


        if (movement_x != 0.0f &&
            movement_y != 0.0f) {

            constexpr float inverse_sqrt_two =
                0.70710678118f;

            movement_x *=
                inverse_sqrt_two;

            movement_y *=
                inverse_sqrt_two;
        }


        player_x +=
            movement_x *
            player_speed;

        player_y +=
            movement_y *
            player_speed;


        std::fill(
            buffer.begin(),
            buffer.end(),
            black);


        const float half_view_width_tiles =
            static_cast<float>(width) /
            (2.0f * pixels_per_tile);

        const float half_view_height_tiles =
            static_cast<float>(height) /
            (2.0f * pixels_per_tile);


        const int min_chunk_x =
            world_to_chunk(
                player_x -
                half_view_width_tiles) - 1;

        const int max_chunk_x =
            world_to_chunk(
                player_x +
                half_view_width_tiles) + 1;

        const int min_chunk_y =
            world_to_chunk(
                player_y -
                half_view_height_tiles) - 1;

        const int max_chunk_y =
            world_to_chunk(
                player_y +
                half_view_height_tiles) + 1;


        for (int chunk_y = min_chunk_y;
             chunk_y <= max_chunk_y;
             ++chunk_y) {

            for (int chunk_x = min_chunk_x;
                 chunk_x <= max_chunk_x;
                 ++chunk_x) {

                const std::pair<int, int> key{
                    chunk_x,
                    chunk_y
                };


                auto chunk_iterator =
                    grid.find(key);

                if (chunk_iterator ==
                    grid.end()) {

                    region_distribution(
                        world_seed,
                        chunk_x,
                        chunk_y,
                        grid);

                    chunk_iterator =
                        grid.find(key);

                    if (chunk_iterator ==
                        grid.end()) {

                        continue;
                    }
                }


                const RegionLocationsChunk& chunk =
                    chunk_iterator->second;


                const std::size_t point_count =
                    std::min({
                        chunk.regions_x.size(),
                        chunk.regions_y.size(),
                        chunk.region_directions.size(),
                        chunk.region_curves.size()
                    });


                for (std::size_t i = 0;
                     i < point_count;
                     ++i) {

                    const std::uint32_t curve =
                        chunk.region_curves[i];


                    const std::uint8_t start_position =
                        get_curve_start(curve);

                    const std::uint8_t point_count_active =
                        get_curve_point_count(curve);


                    if (point_count_active == 0) {
                        continue;
                    }


                    const int world_start_x =
                        chunk_x * chunk_side +
                        static_cast<int>(
                            chunk.regions_x[i]) -
                        point_area_offset;

                    const int world_start_y =
                        chunk_y * chunk_side +
                        static_cast<int>(
                            chunk.regions_y[i]) -
                        point_area_offset;


                    const DecodedCurvePoints decoded =
                        decode_curve_points(
                            world_start_x,
                            world_start_y,
                            curve);


                    const std::uint32_t color =
                        curve_color(
                            chunk_x,
                            chunk_y,
                            i);


                    const int segment_end =
                        static_cast<int>(
                            start_position) +
                        static_cast<int>(
                            point_count_active) -
                        1;


                    for (int p =
                             static_cast<int>(
                                 start_position);
                         p < segment_end;
                         ++p) {

                        const int screen_x0 =
                            width / 2 +
                            static_cast<int>(
                                std::lround(
                                    (static_cast<float>(
                                         decoded.x[p]) -
                                     player_x) *
                                    pixels_per_tile));

                        const int screen_y0 =
                            height / 2 +
                            static_cast<int>(
                                std::lround(
                                    (static_cast<float>(
                                         decoded.y[p]) -
                                     player_y) *
                                    pixels_per_tile));

                        const int screen_x1 =
                            width / 2 +
                            static_cast<int>(
                                std::lround(
                                    (static_cast<float>(
                                         decoded.x[p + 1]) -
                                     player_x) *
                                    pixels_per_tile));

                        const int screen_y1 =
                            height / 2 +
                            static_cast<int>(
                                std::lround(
                                    (static_cast<float>(
                                         decoded.y[p + 1]) -
                                     player_y) *
                                    pixels_per_tile));


                        draw_thick_line(
                            buffer,
                            screen_x0,
                            screen_y0,
                            screen_x1,
                            screen_y1,
                            curve_line_radius,
                            color);
                    }


                    for (int p =
                             static_cast<int>(
                                 start_position);
                         p <= segment_end;
                         ++p) {

                        const int screen_x =
                            width / 2 +
                            static_cast<int>(
                                std::lround(
                                    (static_cast<float>(
                                         decoded.x[p]) -
                                     player_x) *
                                    pixels_per_tile));

                        const int screen_y =
                            height / 2 +
                            static_cast<int>(
                                std::lround(
                                    (static_cast<float>(
                                         decoded.y[p]) -
                                     player_y) *
                                    pixels_per_tile));


                        draw_square(
                            buffer,
                            screen_x,
                            screen_y,
                            curve_point_radius,
                            color);
                    }


                    if (start_position == 0) {

                        const float origin_world_x =
                            static_cast<float>(
                                world_start_x);

                        const float origin_world_y =
                            static_cast<float>(
                                world_start_y);


                        draw_square(
                            buffer,

                            width / 2 +
                            static_cast<int>(
                                std::lround(
                                    (origin_world_x -
                                     player_x) *
                                    pixels_per_tile)),

                            height / 2 +
                            static_cast<int>(
                                std::lround(
                                    (origin_world_y -
                                     player_y) *
                                    pixels_per_tile)),

                            origin_point_radius,
                            color);


                        draw_direction_arrow(
                            buffer,
                            origin_world_x,
                            origin_world_y,
                            chunk.region_directions[i],
                            player_x,
                            player_y,
                            color);
                    }
                }
            }
        }


        // Draw chunk borders after the curves so the borders
        // are always visible during debugging.
        draw_chunk_boundaries(
            buffer,
            player_x,
            player_y);


        draw_square(
            buffer,
            width / 2,
            height / 2,
            player_point_radius,
            white);


        if (mfb_update(
                window,
                buffer.data()) < 0) {

            break;
        }
    }


    return 0;
}