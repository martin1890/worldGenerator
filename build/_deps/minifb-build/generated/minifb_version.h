#pragma once

#include <stdint.h>

#define MINIFB_VERSION_MAJOR 0
#define MINIFB_VERSION_MINOR 10
#define MINIFB_VERSION_PATCH 1
#define MINIFB_VERSION_STRING "0.10.1"

#define MINIFB_COMMITS_SINCE_TAG 3
#define MINIFB_COMMIT_COUNT 422
#define MINIFB_GIT_SHA "2ca5686"
#define MINIFB_GIT_DIRTY 0

#define MINIFB_VERSION_NUMERIC ( ((uint64_t)MINIFB_VERSION_MAJOR << 32) | ((uint64_t)MINIFB_VERSION_MINOR << 16) | (uint64_t)MINIFB_VERSION_PATCH )
#define MINIFB_VERSION_GET_MAJOR(v) ( (uint16_t)(((uint64_t)(v) >> 32) & 0xFFFFu) )
#define MINIFB_VERSION_GET_MINOR(v) ( (uint16_t)(((uint64_t)(v) >> 16) & 0xFFFFu) )
#define MINIFB_VERSION_GET_PATCH(v) ( (uint16_t)(((uint64_t)(v)      ) & 0xFFFFu) )
