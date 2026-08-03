@echo off
"C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\cmake\\3.22.1\\bin\\cmake.exe" ^
  "-HC:\\zerocopy-infer\\android\\AndroidProject\\app\\src\\main\\cpp" ^
  "-DCMAKE_SYSTEM_NAME=Android" ^
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" ^
  "-DCMAKE_SYSTEM_VERSION=26" ^
  "-DANDROID_PLATFORM=android-26" ^
  "-DANDROID_ABI=arm64-v8a" ^
  "-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a" ^
  "-DANDROID_NDK=C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\ndk\\25.1.8937393" ^
  "-DCMAKE_ANDROID_NDK=C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\ndk\\25.1.8937393" ^
  "-DCMAKE_TOOLCHAIN_FILE=C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\ndk\\25.1.8937393\\build\\cmake\\android.toolchain.cmake" ^
  "-DCMAKE_MAKE_PROGRAM=C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\cmake\\3.22.1\\bin\\ninja.exe" ^
  "-DCMAKE_CXX_FLAGS=-std=c++23 -O3 -flto -march=armv8-a+simd" ^
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=C:\\zerocopy-infer\\android\\AndroidProject\\app\\build\\intermediates\\cxx\\Debug\\5l2m5g33\\obj\\arm64-v8a" ^
  "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=C:\\zerocopy-infer\\android\\AndroidProject\\app\\build\\intermediates\\cxx\\Debug\\5l2m5g33\\obj\\arm64-v8a" ^
  "-DCMAKE_BUILD_TYPE=Debug" ^
  "-BC:\\zerocopy-infer\\android\\AndroidProject\\app\\.cxx\\Debug\\5l2m5g33\\arm64-v8a" ^
  -GNinja ^
  "-DANDROID_STL=c++_shared"
