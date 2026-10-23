# W4② C++ 推理（里程碑 1：反序列化 + 单帧推理 + 数值对拍）
#
# 生成器用法（不要写死路径，CI/别的机器上传参即可）：
#   cmake -S cpp -B cpp/build -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DTENSORRT_INCLUDE_DIR="C:/Users/11646/Documents/ChatGPT/简历/_vendor/TensorRT-v10.16/include" \
#     -DTENSORRT_LIBRARY_DIR=".../cpp/trt_lib" \
#     -DCUDA_INCLUDE_DIR="W:/Anaconda/envs/induvision/Library/include" \
#     -DCUDA_LIB_DIR="W:/Anaconda/envs/induvision/Library/lib/x64"
#
# 为什么不用 find_package(CUDAToolkit) / find_package(TensorRT)：
# cudart 来自 conda env（目录布局与官方 Toolkit 不同），TRT 只有头 + 手工合成的
# .lib（没有 CMake config）。显式传四个路径比让 find 模块猜 conda 布局更诚实。
cmake_minimum_required(VERSION 3.24)
project(induvision_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TENSORRT_INCLUDE_DIR "" CACHE PATH "TensorRT 头目录（_vendor/TensorRT-v10.16/include）")
set(TENSORRT_LIBRARY_DIR "" CACHE PATH "TRT import lib 目录（S5(b) 合成的 .lib 所在）")
set(CUDA_INCLUDE_DIR "" CACHE PATH "cudart 头目录（conda env 的 Library/include）")
set(CUDA_LIB_DIR "" CACHE PATH "cudart.lib 目录（conda env 的 Library/lib/x64）")

foreach(v TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY_DIR CUDA_INCLUDE_DIR CUDA_LIB_DIR)
  if("${${v}}" STREQUAL "")
    message(FATAL_ERROR "缺 -D${v}=<path>（见本文件头部的注释）")
  endif()
endforeach()

add_executable(infer_one
  src/infer_one.cpp
  src/engine_header.cpp)

# 源码里有中文注释：MSVC 默认按本地码表（GBK）读，/utf-8 不加就会咬人
target_compile_options(infer_one PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/utf-8 /W4 /O2>)

target_include_directories(infer_one PRIVATE
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIR})

# TRT 是 C++ 类接口 + __declspec(dllimport)，必须链接 import lib；
# LoadLibrary + GetProcAddress 那条路走不通（cpp_infer.md §5）。
target_link_directories(infer_one PRIVATE ${TENSORRT_LIBRARY_DIR} ${CUDA_LIB_DIR})
target_link_libraries(infer_one PRIVATE nvinfer.lib cudart.lib)

# ---------------------------------------------------------------- bench（里程碑 2/3）
# preproc.cu 由 build.bat 里的 nvcc 单独编成 preproc.obj 再进链接 —— 不开 CMake
# 的 CUDA language：conda 布局下 find_package(CUDAToolkit) 猜目录不如显式传参诚实，
# 而且 nvcc 只负责这一个翻译单元（--use_local_env 见 build.bat）。
set(OPENCV_INCLUDE_DIR "" CACHE PATH "OpenCV 头目录（cppenv 的 Library/include）")
set(OPENCV_LIB_DIR "" CACHE PATH "OpenCV .lib 目录（cppenv 的 Library/lib）")
set(NVCC_OBJ "${CMAKE_BINARY_DIR}/preproc.obj")

if(EXISTS "${NVCC_OBJ}")
  add_executable(bench
    src/bench.cpp
    src/engine_header.cpp
    src/postproc.cpp
    "${NVCC_OBJ}")
  target_compile_options(bench PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8 /W4 /O2>)
  target_include_directories(bench PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR})
  target_link_directories(bench PRIVATE
    ${TENSORRT_LIBRARY_DIR} ${CUDA_LIB_DIR} ${OPENCV_LIB_DIR})
  # 只用得到 core/imgproc/imgcodecs 三个模块（imdecode + resize）；
  # shell32 是 CommandLineToArgvW（utf8_args，中文参数口径）
  target_link_libraries(bench PRIVATE nvinfer.lib cudart.lib shell32.lib
    opencv_core500.lib opencv_imgproc500.lib opencv_imgcodecs500.lib)
else()
  message(STATUS "未找到 ${NVCC_OBJ}，跳过 bench（build.bat 会先跑 nvcc）")
endif()
