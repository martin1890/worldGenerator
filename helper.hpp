#ifndef HELPER_HPP
#define HELPER_HPP

#include <array>
#include <cstddef>
#include <cstdint>

float hash_random(uint32_t seed, int32_t x, int32_t y);
float new_random(float current);

template <std::size_t N>
class IntQueue {
private:
    std::array<unsigned int, N> data;
    std::size_t head = 0;
    std::size_t tail = 0;
    std::size_t count = 0;

public:
    bool push(unsigned int x) {
        if (count == N) return false;
        data[tail] = x;
        tail = (tail + 1) % N;
        ++count;
        return true;
    }

    bool pop() {
        if (count == 0) return false;
        head = (head + 1) % N;
        --count;
        return true;
    }

    unsigned int front() const {
        return data[head];
    }

    bool empty() const {
        return count == 0;
    }

    bool full() const {
        return count == N;
    }

    std::size_t size() const {
        return count;
    }
};

#endif