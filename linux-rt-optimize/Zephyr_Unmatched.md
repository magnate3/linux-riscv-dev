# Zephyr RTOS on HiFive Unmatched

## 前置依赖

- Host: Arch Linux
- Board: HiFive Unmatched
- microUSB 数据线一条（HiFive Unmatched 原厂已附带一根）
- ATX 电源一个，推荐 250W 以上

> 若使用的不是 Arch 而是其他发行版，请参照 Zephyr [官方文档](https://docs.zephyrproject.org/latest/develop/getting_started/index.html) 进行环境搭建。

### 依赖包安装

需要已安装 `paru` 或 `yay` 等 AUR Helper。

```bash
# yay -S python-west zephyr-sdk openocd
paru -S python-west zephyr-sdk openocd pyocd
```

## 编译示例程序

```bash
cp /usr/share/zephyr-sdk/zephyrrc ~/.zephyrrc
# 若不是从 AUR 安装，注意 SDK 路径
sudo cp /opt/zephyr-sdk/sysroots/x86_64-pokysdk-linux/usr/share/openocd/contrib/60-openocd.rules /etc/udev/rules.d/
sudo udevadm control --reload
python3 -m venv ~/zephyrproject/.venv
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
source ~/zephyrproject/.venv/bin/activate
west init ~/zephyrproject/
cd ~/zephyrproject/
west update
west zephyr-export
pip install -r ~/zephyrproject/zephyr/scripts/requirements.txt
west build -p always -b hifive_unmatched samples/hello_world
```

## 运行示例程序

将 micro USB 线缆连接至 Host PC，并给 HiFive Unmatched 上电开机。

新启动一个终端，minicom/screen 等工具打开串口：

```bash
sudo minicom -D /dev/ttyUSB1 -b 115200
```

> 不需要插入 microSD 卡。如果插入了 microSD 卡，请在进入 U-Boot 之后按任意键手动打断启动流程。

新启动一个终端，启动 openocd：

```bash
openocd -c 'bindto 0.0.0.0' \
        -f ~/zephyrproject/zephyr/boards/riscv/hifive_unmatched/support/openocd_hifive_unmatched.cfg
```

新启动一个终端，执行 gdb 远程调试：

```bash
# 若不是从 AUR 安装，注意 SDK 下的 gdb 路径
/opt/zephyr-sdk/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gdb ~/zephyrproject/zephyr/build/zephyr/zephyr.elf \
--batch -ex 'target extended-remote localhost:3333' \
-ex 'load' -ex 'monitor resume' -ex 'monitor shutdown' -ex 'quit'
```

运行输出：

```
*** Booting Zephyr OS build v3.6.0-rc3-8-ga48c958c8fb8 ***
Hello World! hifive_unmatched
```

![](./images/zephyr_unmatched.png)

## 参考文档 / Credits

- [SiFive HiFive Unmatched - Zephyr Project](https://docs.zephyrproject.org/latest/boards/riscv/hifive_unmatched/doc/index.html)
- [Getting Started Guide - Zephyr Project Documentation](https://docs.zephyrproject.org/latest/develop/getting_started/index.html)
- [Getting Started with Zephyr RTOS v1.13.0 On RISC-V - SiFive Blog](https://www.sifive.cn/blog/getting-started-with-zephyr-rtos-v1.13.0-on-risc-v)