# Main CMakeFile.txt
 
# Minimal version of CMake
cmake_minimum_required (VERSION 2.6)
 
# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Debug' as none was specified.")
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()
 
# Define project name
project (Client-Server)
 
# Source code for server		
set (server_src server.c)
 
# Source code for client		
set (client_src client.c)
 
# Compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_C_FLAGS "-D_REETRANT -Wall -Wextra -pedantic -Wno-long-long")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   	    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb -O0")
	elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
	    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")
	endif ()
endif (CMAKE_COMPILER_IS_GNUCC)
 
# Set up verse server executable
add_executable (server ${server_src})
 
# Set up verse server executable
add_executable (client ${client_src})
