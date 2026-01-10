cmake_minimum_required(VERSION 3.16)
project(proj2_simd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # helps IntelliSense in VS Code

# --- Library with your implementation ---
add_library(alpha_hist_lib
    src/stb_impl.cpp
    # add your source files here as you create them, e.g.:
    src/io.cpp
    src/utils.cpp
    src/color_lut.cpp
    src/color_convert.cpp
    src/alpha_blend.cpp
    src/histogram.cpp
)

target_include_directories(alpha_hist_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Good warning flags (GCC/Clang)
target_compile_options(alpha_hist_lib PRIVATE
    -Wall -Wextra -Wpedantic -mavx2 -mfma
)

# --- App executable ---
add_executable(alpha_hist_app
    apps/main.cpp
)
target_link_libraries(alpha_hist_app PRIVATE alpha_hist_lib)

# --- Synthetic generator tool ---
add_executable(gen_synth
    apps/gen_synth.cpp
)
target_link_libraries(gen_synth PRIVATE alpha_hist_lib)

# --- Alpha blend tests ---
add_executable(test_alpha_blend
    tests/test_alpha_blend_main.cpp
    tests/test_alpha_blend.cpp
)
target_link_libraries(test_alpha_blend PRIVATE alpha_hist_lib)
target_include_directories(test_alpha_blend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# --- Histogram tests ---
add_executable(test_histogram
    tests/test_histogram_main.cpp
    tests/test_histogram.cpp
)
target_link_libraries(test_histogram PRIVATE alpha_hist_lib)
target_include_directories(test_histogram PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
