cmake_minimum_required(VERSION 3.15)
project(5g_traffic_generator)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS " -pthread")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCAP REQUIRED libpcap)
pkg_check_modules(JANSSON REQUIRED jansson)

# Export include/library dirs for subdirs
include_directories(${PCAP_INCLUDE_DIRS} ${JANSSON_INCLUDE_DIRS})
link_directories(${PCAP_LIBRARY_DIRS} ${JANSSON_LIBRARY_DIRS})

# Add subdirectories
add_subdirectory(src)
