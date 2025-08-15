cmake_minimum_required(VERSION 2.8)

option(USE_HOST_MEM "Use hostmemory for data transfer" OFF)

# set the project name
project(gpu_ibv_utils)

# Shared example files
add_library(gpu_ibverbs_tools STATIC
  pingpong.c
  )

# add the executable
if(USE_HOST_MEM)
	add_executable(rc_pingpong rc_pingpong.c)
	target_link_libraries(rc_pingpong LINK_PRIVATE ibverbs gpu_ibverbs_tools)
	add_executable(ud_pingpong ud_pingpong.c)
	target_link_libraries(ud_pingpong LINK_PRIVATE ibverbs gpu_ibverbs_tools)
	add_definitions(-DUSE_HOST_MEM)
else ()
	add_executable(rc_pingpong rc_pingpong.c)
	target_link_libraries(rc_pingpong LINK_PRIVATE ibverbs gpu_ibverbs_tools cuda)
	add_executable(ud_pingpong ud_pingpong.c)
	target_link_libraries(ud_pingpong LINK_PRIVATE ibverbs gpu_ibverbs_tools cuda)
endif()
