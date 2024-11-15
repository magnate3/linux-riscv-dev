

# amp

[amp](https://doc.rvspace.org/VisionFive2/Application_Notes/RT-Thread/])     


## 编译步骤

[编译步骤](https://doc.rvspace.org/VisionFive2/Application_Notes/RT-Thread/VisionFive_2/RT_Thread/configuration.html)   


### Generate Booting SD Card   

[Generate Booting SD Card](https://github.com/starfive-tech/VisionFive2/tree/JH7110_VisionFive2_6.6.y_devel)  

### Build  AMP Image of  SD Card 
First need to download RT-Thread code:  

```
$ git clone -b amp-5.0.2-devel https://github.com/starfive-tech/rt-thread.git rtthread
```

Then download and prepare the toolchain needed for RT-Thread code:   
```
# For Ubuntu 18.04:
$ wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2022.04.12/riscv64-elf-ubuntu-18.04-nightly-2022.04.12-nightly.tar.gz
$ sudo tar xf riscv64-elf-ubuntu-18.04-nightly-2022.04.12-nightly.tar.gz -C /opt/
$ /opt/riscv/bin/riscv64-unknown-elf-gcc --version
riscv64-unknown-elf-gcc (g5964b5cd727) 11.1.0
```
Generate rtthread amp sdcard image:  
```
$ make -j$(nproc)
$ make ampuboot_fit  # build amp uboot image
$ make buildroot_rootfs -j$(nproc)
$ make img
$ make amp_img       # generate sdcard img
```
