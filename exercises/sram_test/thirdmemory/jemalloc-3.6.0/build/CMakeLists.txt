set_property(GLOBAL PROPERTY USE_FOLDERS On)
cmake_minimum_required(VERSION 2.6)
PROJECT( jemalloc )

include_directories( ../include )

set(
	JEMALLOC_SRC  
	../src/jemalloc.c ../src/arena.c 
	../src/atomic.c ../src/base.c ../src/bitmap.c 
	../src/chunk.c ../src/chunk_dss.c 
	../src/chunk_mmap.c ../src/ckh.c ../src/ctl.c 
	../src/extent.c ../src/hash.c ../src/huge.c 
	../src/mb.c ../src/mutex.c ../src/prof.c 
	../src/quarantine.c ../src/rtree.c ../src/stats.c 
	../src/tcache.c ../src/util.c ../src/tsd.c
)

add_definitions(-D_DEBUG)
set( CMAKE_C_FLAGS " -std=gnu99 -Wall -pipe -g3 -fvisibility=hidden -O3 -funroll-loops -fPIC -DPIC -D_GNU_SOURCE -D_REENTRANT")
set( CMAKE_C_FLAGS_DEBUG " -std=gnu99 -Wall -pipe -g3 -fvisibility=hidden -O3 -funroll-loops -fPIC -DPIC -D_GNU_SOURCE -D_REENTRANT")
set( CMAKE_C_FLAGS_RELEASE " -std=gnu99 -Wall -pipe -g3 -fvisibility=hidden -O3 -funroll-loops -fPIC -DPIC -D_GNU_SOURCE -D_REENTRANT")

ADD_LIBRARY( ${PROJECT_NAME} SHARED ${JEMALLOC_SRC})

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} pthread)

message( STATUS "jemalloc SHARED_LIBS: " ${SHARED_LIBS} )
