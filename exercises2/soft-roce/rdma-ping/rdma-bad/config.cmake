if (NOT DEFINED SERVER)
    message(FATAL_ERROR "SERVER variable must be defined.")
endif ()
add_definitions(-DSERVER=${SERVER})

if (SERVER)
    message(STATUS "Compiling for the server machine")
else ()
    message(STATUS "Compiling for the client machine")
endif ()

if (NOT DEFINED DEBUG)
    set(DEBUG 0)
endif ()
add_definitions(-DDEBUG=${DEBUG})

if (DEBUG)
    message(STATUS "Compiling with debug information")
endif ()

