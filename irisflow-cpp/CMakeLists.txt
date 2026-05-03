cmake_minimum_required(VERSION 3.14)
project(IrisFlow VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ONNX Runtime (assuming it's installed system-wide or via vcpkg)
# A simple way for now is to use pkg-config or standard find_package if available.
# For now, we will add an interface library if find_package fails or just specify it manually.
# In a real setup, we might need FetchContent or vcpkg. Let's assume vcpkg or system install.
find_package(onnxruntime QUIET)

include_directories(include)

add_executable(irisflow
    src/main.cpp
    src/camera/WebcamCapture.cpp
    src/inference/ONNXEngine.cpp
)

target_include_directories(irisflow PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(irisflow PRIVATE ${OpenCV_LIBS})

if(onnxruntime_FOUND)
    target_link_libraries(irisflow PRIVATE onnxruntime)
endif()
