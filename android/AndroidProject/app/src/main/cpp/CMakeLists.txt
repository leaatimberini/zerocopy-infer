cmake_minimum_required(VERSION 3.22.1)
project("zerocopy_infer_android")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2b -O3 -flto")

if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_definitions(-DENABLE_ARM_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
endif()

add_library(zerocopy_infer SHARED
    zerocopy_infer_jni.cpp
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(zerocopy_infer
    ${log-lib}
    ${android-lib}
)
