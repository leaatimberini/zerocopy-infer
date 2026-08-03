@echo off
"C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\cmake\\3.22.1\\bin\\cmake.exe" ^
  "-HC:\\zerocopy-infer\\android\\AndroidProject\\app\\src\\main\\cpp" ^
  "-DCMAKE_SYSTEM_NAME=Android" ^
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" ^
  "-DCMAKE_SYSTEM_VERSION=26" ^
  "-DANDROID_PLATFORM=android-26" ^
  "-DANDROID_ABI=x86_64" ^
  "-DCMAKE_ANDROID_ARCH_ABI=x86_64" ^
  "-DANDROID_NDK=C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\ndk\\28.2.13676358" ^
  "-DCMAKE_ANDROID_NDK=C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\ndk\\28.2.13676358" ^
  "-DCMAKE_TOOLCHAIN_FILE=C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\ndk\\28.2.13676358\\build\\cmake\\android.toolchain.cmake" ^
  "-DCMAKE_MAKE_PROGRAM=C:\\Users\\timbe\\AppData\\Local\\Android\\Sdk\\cmake\\3.22.1\\bin\\ninja.exe" ^
  "-DCMAKE_CXX_FLAGS=-std=c++23 -O3 -flto -march=armv8-a+simd" ^
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=C:\\zerocopy-infer\\android\\AndroidProject\\app\\build\\intermediates\\cxx\\Debug\\20h1373m\\obj\\x86_64" ^
  "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=C:\\zerocopy-infer\\android\\AndroidProject\\app\\build\\intermediates\\cxx\\Debug\\20h1373m\\obj\\x86_64" ^
  "-DCMAKE_BUILD_TYPE=Debug" ^
  "-BC:\\zerocopy-infer\\android\\AndroidProject\\app\\.cxx\\Debug\\20h1373m\\x86_64" ^
  -GNinja ^
  "-DANDROID_STL=c++_shared"
