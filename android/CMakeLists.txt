# Android NDK CMakeLists.txt for ZeroCopy-Infer
# Target Architecture: arm64-v8a (ARM NEON / C++23)
# Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina)

cmake_minimum_required(VERSION 3.22.1)
project("zerocopy_infer_android")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable ARM NEON vector acceleration
if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_definitions(-DENABLE_ARM_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -march=armv8-a+simd")
endif()

# Shared library output: libzerocopy_infer.so
add_library(zerocopy_infer SHARED
    zerocopy_infer_jni.cpp
    ../cpp/zerocopy_infer.cpp
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(zerocopy_infer
    ${log-lib}
    ${android-lib}
)
