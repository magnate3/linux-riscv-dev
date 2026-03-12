@echo off
cd /d "C:\Users\fabia\Projects\kv-compact"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
if not exist "build-full\build.ninja" (
    echo === Configuring === >> build.log 2>&1
    cmake -G Ninja -S . -B build-full -DLLAMA_CPP_DIR=C:\Users\fabia\Projects\llama.cpp\llama-flash-attn -DCMAKE_BUILD_TYPE=Release >> build.log 2>&1
    if errorlevel 1 (
        echo CONFIG FAILED >> build.log
        exit /b 1
    )
)
echo === Building === >> build.log 2>&1
cmake --build build-full --target llama-kv-compact test-kv-compact-e2e >> build.log 2>&1
if errorlevel 1 (
    echo BUILD FAILED >> build.log
    exit /b 1
)
echo === Build OK === >> build.log
