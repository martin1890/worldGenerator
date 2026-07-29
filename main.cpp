#include <MiniFB.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "region_distribution.hpp"

constexpr int width = 800;
constexpr int height = 600;
constexpr uint32_t world_seed = 12345u;

constexpr float player_speed = 0.18f;
constexpr int pixels_per_tile = 12;
constexpr int region_point_radius = 2;
constexpr int player_point_radius = 4;

constexpr uint32_t black = 0x00000000u;
constexpr uint32_t white = 0x00FFFFFFu;
constexpr uint32_t red = 0x00FF0000u;

using RegionGrid =
    std::unordered_map<std::pair<int, int>, RegionLocationsChunk, PairHash>;

void draw_square(
    std::vector<uint32_t>& buffer,
    int center_x,
    int center_y,
    int radius,
    uint32_t color)
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

int world_to_chunk(float world_coordinate)
{
    return static_cast<int>(
        std::floor(world_coordinate / static_cast<float>(side)));
}

int main()
{
    std::vector<uint32_t> buffer(width * height, black);
    mfb_window* window = mfb_open("Region viewer", width, height);

    if (window == nullptr) {
        return 1;
    }

    RegionGrid grid;

    float player_x = 0.0f;
    float player_y = 0.0f;

    mfb_set_target_fps(60);

    while (mfb_wait_sync(window)) {
        const uint8_t* keys = mfb_get_key_buffer(window);

        float movement_x = 0.0f;
        float movement_y = 0.0f;

        if (keys[KB_KEY_W]) movement_y -= 1.0f;
        if (keys[KB_KEY_S]) movement_y += 1.0f;
        if (keys[KB_KEY_A]) movement_x -= 1.0f;
        if (keys[KB_KEY_D]) movement_x += 1.0f;

        // Prevent diagonal movement from being faster.
        if (movement_x != 0.0f && movement_y != 0.0f) {
            constexpr float inverse_sqrt_two = 0.70710678118f;
            movement_x *= inverse_sqrt_two;
            movement_y *= inverse_sqrt_two;
        }

        player_x += movement_x * player_speed;
        player_y += movement_y * player_speed;

        // Clear the entire screen to black.
        std::fill(buffer.begin(), buffer.end(), black);

        const float half_view_width_tiles =
            static_cast<float>(width) / (2.0f * pixels_per_tile);
        const float half_view_height_tiles =
            static_cast<float>(height) / (2.0f * pixels_per_tile);

        const int min_chunk_x =
            world_to_chunk(player_x - half_view_width_tiles) - 1;
        const int max_chunk_x =
            world_to_chunk(player_x + half_view_width_tiles) + 1;
        const int min_chunk_y =
            world_to_chunk(player_y - half_view_height_tiles) - 1;
        const int max_chunk_y =
            world_to_chunk(player_y + half_view_height_tiles) + 1;

        for (int chunk_y = min_chunk_y; chunk_y <= max_chunk_y; ++chunk_y) {
            for (int chunk_x = min_chunk_x; chunk_x <= max_chunk_x; ++chunk_x) {
                const std::pair<int, int> key{chunk_x, chunk_y};

                if (grid.find(key) == grid.end()) {
                    region_distribution(
                        world_seed,
                        chunk_x,
                        chunk_y,
                        grid);
                }

                const RegionLocationsChunk& chunk = grid.at(key);

                for (const Points& point : chunk.regions) {
                    const float world_x =
                        static_cast<float>(chunk_x * side + point.x);
                    const float world_y =
                        static_cast<float>(chunk_y * side + point.y);

                    const int screen_x = width / 2 + static_cast<int>(
                        std::lround(
                            (world_x - player_x) * pixels_per_tile));

                    const int screen_y = height / 2 + static_cast<int>(
                        std::lround(
                            (world_y - player_y) * pixels_per_tile));

                    draw_square(
                        buffer,
                        screen_x,
                        screen_y,
                        region_point_radius,
                        white);
                }
            }
        }

        // The player stays fixed at the center of the screen.
        draw_square(
            buffer,
            width / 2,
            height / 2,
            player_point_radius,
            red);

        if (mfb_update(window, buffer.data()) < 0) {
            break;
        }
    }

    return 0;
}