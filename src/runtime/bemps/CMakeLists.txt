add_library(bemps SHARED
    bemps.cpp
)

include_directories(/usr/local/cuda/include)

find_library(LIBCUDART cudart HINTS /usr/local/cuda/lib64)
find_library(LIBRT rt HINTS /usr/local/cuda/lib64)
message("libcudart: " ${LIBCUDART})
message("librt: " ${LIBRT})

set(CMAKE_CXX_FLAGS "-g")
target_link_libraries(bemps ${LIBCUDART} ${LIBRT})