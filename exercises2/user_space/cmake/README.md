

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


#   鲲鹏 编译安装CMake
下载CMake源码并解压。
```
wget https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz
tar -zxf cmake-3.5.2.tar.gz
```
编译安装。
```
cd cmake-3.5.2
./bootstrap && make -j64 && make install -j64
```
清除系统的Hash，否则可能引用到旧版本CMake。
```
hash -r
```
检查CMake是否安装成功。
```
cmake --version
```
回显如下所示即为安装成功。



## openssl


```
set(CMAKE_USE_OPENSSL OFF)
```
 