#include <MiniFB.h>
#include <vector>

#include "helper.hpp"
#include "poisson_disc.hpp"

const int width = 800;
const int height = 600;

int main() {
    std::vector<uint32_t> buffer(width * height, 0x000000);
    mfb_window* window = mfb_open("Viewer", width, height);

    while (mfb_wait_sync(window)) {

        for (int y = 300; y < 400; ++y) {
            for (int x = 300; x < 400; ++x) {
                buffer[y * width + x] = 0x00FFFFFF;
            }
        }

        if (mfb_update(window, buffer.data()) < 0) {
            break;
        }
    }

    return 0;
}