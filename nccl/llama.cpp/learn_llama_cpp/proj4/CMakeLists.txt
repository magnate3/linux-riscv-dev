#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LLAMA_CPP_DIR /workspace/llama.cpp)
set(LLAMA_CPP_SRC_DIR ${LLAMA_CPP_DIR}/src)
set(LLAMA_CPP_MODEL_SRC_DIR ${LLAMA_CPP_SRC_DIR}/models)
set(LLAMA_CPP_COMMON_SRC_DIR ${LLAMA_CPP_DIR}/common)
set(TARGET debug_kv)
message(STATUS "llama-adapter.cpp src path: ${LLAMA_CPP_SRC_DIR}/llama-adapter.cpp")
message(STATUS "mimo2-iswa.cpp src path: ${LLAMA_CPP_MODEL_SRC_DIR}/mimo2-iswa.cpp")
add_executable(${TARGET} debug_kv.cpp
    ${LLAMA_CPP_SRC_DIR}/llama.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-adapter.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-arch.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-batch.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-chat.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-context.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-cparams.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-grammar.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-graph.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-hparams.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-impl.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-io.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-kv-cache.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-kv-cache-iswa.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-memory-hybrid.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-memory-hybrid-iswa.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-memory-recurrent.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-memory.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-mmap.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-model-loader.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-model-saver.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-model.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-quant.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-sampling.cpp
    ${LLAMA_CPP_SRC_DIR}/llama-vocab.cpp
    ${LLAMA_CPP_SRC_DIR}/unicode.cpp
    ${LLAMA_CPP_SRC_DIR}/unicode-data.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/afmoe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/apertus.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/arcee.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/arctic.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/arwkv7.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/baichuan.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/bailingmoe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/bailingmoe2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/bert.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/bitnet.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/bloom.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/chameleon.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/chatglm.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/codeshell.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/cogvlm.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/cohere2-iswa.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/command-r.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/dbrx.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/deci.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/deepseek.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/deepseek2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/dots1.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/dream.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/ernie4-5-moe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/ernie4-5.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/exaone.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/exaone4.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/exaone-moe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/falcon-h1.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/falcon.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/gemma-embedding.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/gemma.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/gemma2-iswa.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/gemma3.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/gemma3n-iswa.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/glm4-moe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/glm4.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/gpt2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/gptneox.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/granite-hybrid.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/granite.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/grok.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/grovemoe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/hunyuan-dense.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/hunyuan-moe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/internlm2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/jais.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/jamba.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/lfm2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/llada-moe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/llada.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/llama-iswa.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/llama.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/maincoder.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/mamba.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/mimo2-iswa.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/minicpm3.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/minimax-m2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/modern-bert.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/mpt.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/nemotron-h.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/nemotron.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/neo-bert.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/olmo.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/olmo2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/olmoe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/openai-moe-iswa.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/openelm.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/orion.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/pangu-embedded.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/phi2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/phi3.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/plamo.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/plamo2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/plamo3.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/plm.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen2moe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen2vl.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen3.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen3vl.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen3vl-moe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen3moe.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/qwen3next.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/refact.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/rnd1.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/rwkv6-base.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/rwkv6.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/rwkv6qwen2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/rwkv7-base.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/rwkv7.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/seed-oss.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/smallthinker.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/smollm3.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/stablelm.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/starcoder.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/starcoder2.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/t5-dec.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/t5-enc.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/wavtokenizer-dec.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/xverse.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/mistral3.cpp
    ${LLAMA_CPP_MODEL_SRC_DIR}/graph-context-mamba.cpp
    ${LLAMA_CPP_DIR}/build/common/build-info.cpp
    ${LLAMA_CPP_COMMON_SRC_DIR}/log.cpp
    ${LLAMA_CPP_COMMON_SRC_DIR}/sampling.cpp
    ${LLAMA_CPP_COMMON_SRC_DIR}/common.cpp)
set(LLAMA_INCLUDE_DIR ${LLAMA_CPP_DIR}/vendor  ${LLAMA_CPP_DIR}/ggml/include  ${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/src ${LLAMA_CPP_DIR}/common ${LLAMA_CPP_DIR}/tools/mtmd)
target_include_directories(${TARGET} PRIVATE ${LLAMA_INCLUDE_DIR})
target_compile_options(${TARGET} PRIVATE -g)
target_link_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/build/bin  ${LLAMA_CPP_DIR}/build/vendor/cpp-httplib ${LLAMA_CPP_DIR}/build/common )
#target_link_directories(${TARGET} PRIVATE ${LLAMA_CPP_DIR}/build/bin  ${LLAMA_CPP_DIR}/build/vendor/cpp-httplib ${LLAMA_CPP_DIR}/build/common )
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-base  ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-base  common   ${CMAKE_THREAD_LIBS_INIT})
