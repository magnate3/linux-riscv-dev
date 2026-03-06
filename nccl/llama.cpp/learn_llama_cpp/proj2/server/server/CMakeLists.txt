set(CMAKE_CXX_FLAGS "-DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED -DLLAMA_USE_HTTPLIB  ${CMAKE_CXX_FLAGS}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

###  my custom
set(LLAMA_CPP_DIR /workspace/llama.cpp)
function(custom_llama_cpp target)
    set(LLAMA_INCLUDE_DIR ${LLAMA_CPP_DIR}/vendor  ${LLAMA_CPP_DIR}/ggml/include  ${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/common ${LLAMA_CPP_DIR}/tools/mtmd)
    link_directories(${LLAMA_CPP_DIR}/build/bin  ${LLAMA_CPP_DIR}/build/vendor/cpp-httplib ${LLAMA_CPP_DIR}/build/common )
    target_include_directories(${TARGET} PRIVATE ${LLAMA_INCLUDE_DIR})
endfunction()
#########
custom_llama_cpp(${TARGET})
#target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC common mtmd ${CMAKE_THREAD_LIBS_INIT})


# llama-server executable

set(TARGET llama-server)

set(LLAMA_HTTPLIB "ON")
if (NOT LLAMA_HTTPLIB)
    message(FATAL_ERROR "LLAMA_HTTPLIB is OFF, cannot build llama-server. Hint: to skip building server, set -DLLAMA_BUILD_SERVER=OFF")
endif()

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

## my custome
foreach(asset ${PUBLIC_ASSETS})
    set(PROJECT_SOURCE_DIR ${LLAMA_CPP_DIR})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    message(STATUS "xxd.cmake path: ${PROJECT_SOURCE_DIR}/scripts/xxd.cmake")
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()


add_executable(${TARGET} ${TARGET_SRCS})
#target_include_directories(${TARGET} PRIVATE ${LLAMA_INCLUDE_DIR})
custom_llama_cpp(${TARGET})
install(TARGETS ${TARGET} RUNTIME)
#target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
#target_link_libraries(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR/build/server-context ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET ${TARGET} PROPERTY LINK_LIBRARIES ${CMAKE_SOURCE_DIR}/build/libserver-context.a ${LLAMA_CPP_DIR}/build/common/libcommon.a ${LLAMA_CPP_DIR}/build/vendor/cpp-httplib/libcpp-httplib.a /usr/lib/aarch64-linux-gnu/libssl.so /usr/lib/aarch64-linux-gnu/libcrypto.so ${LLAMA_CPP_DIR}/build/bin/libmtmd.so.0.0.7941 ${LLAMA_CPP_DIR}/build/bin/libllama.so.0.0.7941 ${LLAMA_CPP_DIR}/build/bin/libggml.so.0.9.5 ${LLAMA_CPP_DIR}/build/bin/libggml-cpu.so.0.9.5 ${LLAMA_CPP_DIR}/build/bin/libggml-base.so.0.9.5)
#set_property(TARGET ${target} PROPERTY LINK_LIBRARIES ${LLAMA_CPP_DIR}/build/common/libcommon.a ${LLAMA_CPP_DIR}/build/bin/libmtmd.so.0.0.7941 ${LLAMA_CPP_DIR}/build/bin/libllama.so.0.0.7941 ${LLAMA_CPP_DIR}/build/bin/libggml.so.0.9.5 ${LLAMA_CPP_DIR}/build/bin/libggml-cpu.so.0.9.5 ${LLAMA_CPP_DIR}/build/bin/libggml-base.so.0.9.5 ${LLAMA_CPP_DIR}/build/vendor/cpp-httplib/libcpp-httplib.a /usr/lib/aarch64-linux-gnu/libssl.so /usr/lib/aarch64-linux-gnu/libcrypto.so)

#target_link_libraries(${TARGET} PRIVATE server-context ${CMAKE_THREAD_LIBS_INIT})
if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)
