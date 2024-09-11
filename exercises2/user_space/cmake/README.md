

# 交叉编译


+ 删除旧的   
```
rm -rf  Makefile   CMakeCache.txt   CMakeFiles  cmake_install.cmake  
```

+ 创建toolchain.cmake     

```
root@ubuntu:~/SOEM/build# cat ../toolchain.cmake 
# 指定目标系统
set(CMAKE_SYSTEM_NAME Linux)
# 指定目标平台
set(CMAKE_SYSTEM_PROCESSOR arm)
 
# 指定交叉编译工具链的根路径
set(CROSS_CHAIN_PATH  /usr/bin/)
# 指定C编译器
set(CMAKE_C_COMPILER "${CROSS_CHAIN_PATH}/riscv64-linux-gnu-gcc")
# 指定C++编译器
set(CMAKE_CXX_COMPILER "${CROSS_CHAIN_PATH}/riscv64-linux-gnu-c++")
root@ubuntu:~/SOEM/build# 
```

+ cmake ..    
 
```
root@ubuntu:~/SOEM/build# cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake
-- The C compiler identification is GNU 8.4.0
-- Check for working C compiler: /usr/bin//riscv64-linux-gnu-gcc
-- Check for working C compiler: /usr/bin//riscv64-linux-gnu-gcc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- OS is linux
-- LIB_DIR: lib
-- Configuring done
-- Generating done
-- Build files have been written to: /root/SOEM/build
root@ubuntu:~/SOEM/build# cd -
/root/SOEM
```

+ make   


```
root@ubuntu:~/SOEM/build# make -j64
Scanning dependencies of target soem
[ 42%] Building C object CMakeFiles/soem.dir/osal/linux/osal.c.o
[ 42%] Building C object CMakeFiles/soem.dir/soem/ethercatconfig.c.o
[ 42%] Building C object CMakeFiles/soem.dir/soem/ethercatfoe.c.o
```
 