cmake_minimum_required(VERSION 3.10)
project(openpylivoxv2)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)


execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)


if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()


if(UNIX AND NOT APPLE)
  
  find_package(Threads REQUIRED)
  
endif()


if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Livox-SDK2/CMakeLists.txt")
    message(FATAL_ERROR "Livox-SDK2 not found. Please make sure it exists in the project directory.")
endif()


option(USE_PREBUILT_LIVOX_SDK "Use pre-built Livox SDK2 library" OFF)

if(USE_PREBUILT_LIVOX_SDK)
    
    find_library(LIVOX_SDK_LIBRARY
        NAMES livox_lidar_sdk_static
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/Livox-SDK2/build/lib"
              "${CMAKE_CURRENT_SOURCE_DIR}/Livox-SDK2/build"
              "${CMAKE_CURRENT_SOURCE_DIR}/Livox-SDK2/lib"
        NO_DEFAULT_PATH
    )
    
    if(NOT LIVOX_SDK_LIBRARY)
        message(FATAL_ERROR "Prebuilt Livox SDK2 library not found. Please build it first or set USE_PREBUILT_LIVOX_SDK=OFF.")
    endif()
    
    message(STATUS "Using prebuilt Livox SDK2 library: ${LIVOX_SDK_LIBRARY}")
else()
    
    message(STATUS "Building Livox SDK2 as part of this project")
    add_subdirectory(Livox-SDK2)
endif()


pybind11_add_module(openpylivoxv2 openpylivoxv2.cpp)


target_include_directories(openpylivoxv2 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Livox-SDK2/include
    ${Python_NumPy_INCLUDE_DIRS}
)


if(USE_PREBUILT_LIVOX_SDK)
    target_link_libraries(openpylivoxv2 PRIVATE ${LIVOX_SDK_LIBRARY})
else()
    target_link_libraries(openpylivoxv2 PRIVATE livox_lidar_sdk_static)
endif()


if(UNIX AND NOT APPLE)
    target_link_libraries(openpylivoxv2 PRIVATE Threads::Threads)
endif()


if(WIN32)
    target_link_libraries(openpylivoxv2 PRIVATE ws2_32)
endif()


set_target_properties(openpylivoxv2 PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  
)


install(TARGETS openpylivoxv2 DESTINATION .)


add_custom_target(build_instructions
    COMMAND ${CMAKE_COMMAND} -E echo "===== Build Instructions for Livox SDK2 Python Wrapper ====="
    COMMAND ${CMAKE_COMMAND} -E echo "Method 1: Build everything together (Recommended)"
    COMMAND ${CMAKE_COMMAND} -E echo "1. Create a build directory: mkdir build"
    COMMAND ${CMAKE_COMMAND} -E echo "2. Navigate to it: cd build"
    COMMAND ${CMAKE_COMMAND} -E echo "3. Configure: cmake .."
    COMMAND ${CMAKE_COMMAND} -E echo "4. Build: cmake --build . --config Release"
    COMMAND ${CMAKE_COMMAND} -E echo "5. The module will be available in the project root directory"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Method 2: Build Livox SDK2 first, then the wrapper"
    COMMAND ${CMAKE_COMMAND} -E echo "1. Build Livox SDK2:"
    COMMAND ${CMAKE_COMMAND} -E echo "   cd Livox-SDK2"
    COMMAND ${CMAKE_COMMAND} -E echo "   mkdir build && cd build"
    COMMAND ${CMAKE_COMMAND} -E echo "   cmake .."
    COMMAND ${CMAKE_COMMAND} -E echo "   cmake --build . --config Release"
    COMMAND ${CMAKE_COMMAND} -E echo "2. Build the Python wrapper:"
    COMMAND ${CMAKE_COMMAND} -E echo "   cd ../../"
    COMMAND ${CMAKE_COMMAND} -E echo "   mkdir build && cd build"
    COMMAND ${CMAKE_COMMAND} -E echo "   cmake .. -DUSE_PREBUILT_LIVOX_SDK=ON"
    COMMAND ${CMAKE_COMMAND} -E echo "   cmake --build . --config Release"
    COMMAND ${CMAKE_COMMAND} -E echo "3. The module will be available in the project root directory"
    VERBATIM
)

message(STATUS "")
message(STATUS "========== Livox SDK2 Python Wrapper Configuration ==========")
message(STATUS "Python Executable: ${Python_EXECUTABLE}")
message(STATUS "Python Version: ${Python_VERSION}")
message(STATUS "Python Include Dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "NumPy Include Dirs: ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "pybind11 Version: ${pybind11_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using Prebuilt Livox SDK: ${USE_PREBUILT_LIVOX_SDK}")
message(STATUS "===========================================================")
message(STATUS "")
