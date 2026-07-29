#include "helper.hpp"
#include <cstring>

float hash_random(uint32_t seed, int32_t x, int32_t y) {
    uint32_t h = seed;

    h ^= static_cast<uint32_t>(x) * 374761393u;
    h ^= static_cast<uint32_t>(y) * 668265263u;

    h ^= h >> 16;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;

    return static_cast<float>(h) / 4294967296.0f;
}

uint16_t hash_random_uint16_t(uint32_t seed, int32_t x, int32_t y)
{
    uint32_t h = seed;

    h ^= static_cast<uint32_t>(x) * 374761393u;
    h ^= static_cast<uint32_t>(y) * 668265263u;

    h ^= h >> 16;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;

    return static_cast<uint16_t>(h);
}

float new_random(float current)
{
    std::uint32_t x;
    std::memcpy(&x, &current, sizeof(x));

    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;

    x = (x >> 9) | 0x3f800000U;

    float result;
    std::memcpy(&result, &x, sizeof(result));

    return result - 1.0f;
}