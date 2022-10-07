add_library(lazy SHARED
    Core.cpp
    Operation.cpp
    Wrappers.cpp
)

include_directories(/usr/local/cuda/include)

find_library(LIBCUDART cudart HINTS /usr/local/cuda/lib64)
message("libcudart: " ${LIBCUDART})

set(CMAKE_CXX_FLAGS "-g")
target_link_libraries(lazy ${LIBCUDART} bemps)

if (DEBUG_LAZY)
    add_definitions(-DDEBUG)
endif()

if (DEBUG_NOLAZY)
    add_definitions(-DNOLAZY)
endif()
