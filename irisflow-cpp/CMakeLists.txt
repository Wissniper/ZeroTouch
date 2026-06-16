cmake_minimum_required(VERSION 3.14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(IrisFlow VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
  list(APPEND CMAKE_PREFIX_PATH
    "/opt/homebrew/opt/opencv"
    "/usr/local/opt/opencv"
  )
endif()

find_package(OpenCV REQUIRED)

add_executable(irisflow
    src/main.cpp
    src/camera/WebcamCapture.cpp
    src/processing/KalmanFilter.cpp
    src/processing/GazeCalibrator.cpp
    src/control/GestureController.cpp
    src/control/OSEventInjector.cpp
)

target_include_directories(irisflow PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(irisflow PRIVATE ${OpenCV_LIBS})

if(APPLE)
    find_library(APP_SERVICES ApplicationServices)
    target_link_libraries(irisflow PRIVATE ${APP_SERVICES})
endif()
