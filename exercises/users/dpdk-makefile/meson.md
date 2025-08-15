
# 查看  meson.build  文件
```
project('DPDK', 'C',
        # Get version number from file.
        # Fallback to "more" for Windows compatibility.
        version: run_command(find_program('cat', 'more'),
            files('VERSION'), check: true).stdout().strip(),
        license: 'BSD',
        default_options: [
            'buildtype=release',
            'default_library=static',
            'warning_level=2',
        ],
        meson_version: '>= 0.53.2'
)
```
+ -include rte_config.h文件   
```
cc -O3 -include rte_config.h -march=native -I/usr/local/include
```
# meson使用简介
```
具体meson有哪些参数，可以通过help查看。
配置meson参数，与cmake类似，-D开头，后续紧跟配置内容
配置meson需要使用meson setup，如果不加setup直接调用meson也可以，但是不建议。

meson -Dexamples=all build 编译所有的示例
meson setup -Dprefix=/home/dpdkinstall build 指定安装目录
meson setup --reconfigure -Dexamples=ethtool build 重新配置
```
# ninja使用简介
```
ninja -C build -j4 多个job同时编译。-C是进入到指定目录执行
ninja install 安装
ninja uninstall 卸载
ninja reconfigure 重新配置
```
# meson.build  cflags

```
cflags += ['-DPF_DRIVER',
    '-DVF_DRIVER',
    '-DINTEGRATED_VF',
    '-DX722_A0_SUPPORT']
```


# 定义宏 
`dpdk_conf.set('RTE_LIBRTE_I40E_INC_VECTOR', 1)`
```
if arch_subdir == 'x86'
        dpdk_conf.set('RTE_LIBRTE_I40E_INC_VECTOR', 1)
        sources += files('i40e_rxtx_vec_sse.c')

        # compile AVX2 version if either:
        # a. we have AVX supported in minimum instruction set baseline
        # b. it's not minimum instruction set, but supported by compiler
        if dpdk_conf.has('RTE_MACHINE_CPUFLAG_AVX2')
                cflags += ['-DCC_AVX2_SUPPORT']
                sources += files('i40e_rxtx_vec_avx2.c')
        elif cc.has_argument('-mavx2')
                cflags += ['-DCC_AVX2_SUPPORT']
                i40e_avx2_lib = static_library('i40e_avx2_lib',
                                'i40e_rxtx_vec_avx2.c',
                                dependencies: [static_rte_ethdev,
                                        static_rte_kvargs, static_rte_hash],
                                include_directories: includes,
                                c_args: [cflags, '-mavx2'])
                objs += i40e_avx2_lib.extract_objects('i40e_rxtx_vec_avx2.c')
        endif
endif
```