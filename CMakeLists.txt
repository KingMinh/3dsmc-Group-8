cmake_minimum_required(VERSION 3.6)


# Build types
set (CMAKE_CONFIGURATION_TYPES "Debug;Release")
# Set default build type to debug
if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)


project(3dsmc)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_SUPPRESS_REGENERATION TRUE)

SET(CMAKE_CXX_STANDARD 17)

# Debug/Release Definitions
if(CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-DENABLE_ASSERTIONS=1)
	add_definitions(-DDEBUG=1)
	set(CMAKE_VERBOSE_MAKEFILE 1)

	if (MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")	
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
	endif()
elseif(CMAKE_BUILD_TYPE MATCHES Release)
	if (MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O3")	
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
	endif()
endif()

# Libraries
set(LIBRARY_DIR ${PROJECT_SOURCE_DIR}/libs CACHE PATH "Path to lib folder")

find_package(Eigen3 3.3 NO_MODULE)
if (NOT TARGET Eigen3::Eigen)
    set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/Eigen/ CACHE PATH "Path to Eigen source folder")
endif (NOT TARGET Eigen3::Eigen)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

# Files
include_directories (${PROJECT_SOURCE_DIR}/src/
                    ${PROJECT_SOURCE_DIR}/src/)

file (GLOB_RECURSE PROJECT_HEADERS   ${PROJECT_SOURCE_DIR}/src/*.hpp
                                     ${PROJECT_SOURCE_DIR}/src/*.h)
file (GLOB_RECURSE PROJECT_SOURCES   ${PROJECT_SOURCE_DIR}/src/*.cpp
                                     ${PROJECT_SOURCE_DIR}/src/*.cxx
                                     ${PROJECT_SOURCE_DIR}/src/*.cc
                                     ${PROJECT_SOURCE_DIR}/src/*.c
                                     ${PROJECT_SOURCE_DIR}/src/*.h
                                     ${PROJECT_SOURCE_DIR}/src/*.hpp)
file (GLOB_RECURSE PROJECT_RESOURCES ${PROJECT_SOURCE_DIR}/res/*.**)

add_executable(3dsmc ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${PROJECT_RESOURCES})
target_link_libraries(3dsmc Eigen3::Eigen ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(3dsmc OpenMP::OpenMP_CXX)
endif()

set_property(TARGET 3dsmc PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")