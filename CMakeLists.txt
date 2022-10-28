project(metal_tests LANGUAGES CXX)
cmake_minimum_required(VERSION 3.13)

include(FetchContent)
FetchContent_Declare(metal
  URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS12_iOS15.zip
)
FetchContent_Populate(metal)

#Metal-cpp Library definition
add_library(Metal-cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/metal_impl/impl_generator.cpp
        )


target_compile_features(Metal-cpp PUBLIC cxx_std_17)

# Metal-cpp headers
target_include_directories(Metal-cpp PUBLIC
  "${metal_SOURCE_DIR}")


# Metal-cpp library (linker)
target_link_libraries(Metal-cpp
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
        )
#myLib
add_library(myLib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib.cpp
        )

target_compile_features(myLib PUBLIC cxx_std_17)

add_dependencies(myLib Metal-cpp)

# Metal cpp headers
target_include_directories(myLib PUBLIC
  "${metal_SOURCE_DIR}")
target_link_libraries(myLib Metal-cpp)

#Executable
add_executable( metal_tests main.cpp )
add_dependencies(metal_tests Metal-cpp)
add_dependencies(metal_tests myLib)
target_link_libraries(metal_tests Metal-cpp myLib)
