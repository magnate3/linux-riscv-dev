


Install DPDK using the MESON build system:
Run the following set of commands from the top-level DPDK directory:
```
meson build
cd build
```
Run meson configure to include kmods:
```
meson configure -Denable_kmods=true
```
```
ninja
sudo ninja install
ldconfig
```

# 配置    PKG_CONFIG_PATH
```
ls /root/dpdk-stable-21.11.8/build/install/
lib  usr
```

```
find ./ -name libdpdk.pc
./build/install/usr/local/lib/x86_64-linux-gnu/pkgconfig/libdpdk.pc
```
更改libdpdk.pc的prefix   

```
prefix=/root/dpdk-stable-21.11.8/build/install/usr/local
```

```
 export PKG_CONFIG_PATH=/root/dpdk-stable-21.11.8/build/install/usr/local/lib/x86_64-linux-gnu/pkgconfig/
# pkg-config --modversion libdpdk
21.11.8
```


```
pkg-config --cflags libdpdk
-I/root/dpdk-stable-21.11.8/build/install/usr/local/include -I/usr/local/include -include rte_config.h -march=native 
```

# 配置 LD_LIBRARY_PATH和LIBRARY_PATH

文中的LIBRARY_PATH是编译时指定的路径。   
LD_LIBRARY_PATH是运行时指定的动态链接库所在目录。   

```
export LD_LIBRARY_PATH=/root/dpdk-stable-21.11.8/build/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/root/dpdk-stable-21.11.8/build/lib:$LIBRARY_PATH
```

# dperf of dpdk 20.11

```
export LD_LIBRARY_PATH=/root/dpdk-stable-21.11.8/build/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/root/dpdk-stable-21.11.8/build/lib:$LIBRARY_PATH
export PKG_CONFIG_PATH=/root/dpdk-stable-21.11.8/build/install/usr/local/lib/x86_64-linux-gnu/pkgconfig/
```


# How to test network bandwidth?
[How to test network bandwidth?](https://dperf.org/doc/html/dperf-faq)   