
https://blog.csdn.net/mybelief321/article/details/8987502
sudo apt-get install minicom
sudo minicom
root
0penBmc

sol.sh
root
onl

ifconfig ma1 192.168.200.2 netmask 255.255.255.0
route add default gw 192.168.200.1

ONL 2.0 安装和编译SDE.pdf

service ssh start

Device /dev/ttyUSB0 is locked.解决办法
cd /var/lock
删除目录下的LOCK


#chongqiwangka
/etc/init.d/networking restart

3.3. 更新ONL依赖库
$ sudo apt-get update
$ sudo apt-get install -y git build-essential python-dev python-setuptools
libboost-all-dev libssl-dev libevent-dev libpcap-dev libusb-1.0-0-dev libcurl4-openssl-dev
 libi2c-dev libJudy-dev libboost-thread-dev


4.7. 4.6
并设置环境
执行安装 sde
./p4studio_build.py --use-profile diags_profile --bsp-path ~/bf-
reference-bsp-8.8.1


export SDE=/root/bf-sde-8.9.1
export SDE_INSTALL=/root/bf-sde-8.9.1/install
export PATH=$PATH:$SDE_INSTALL/bin

./install/bin/bf_kdrv_mod_load $SDE_INSTALL/
ls /dev/bf0

ssh root@192.168.200.2
onl

#compile ***.p4
cd ./bf-sde-8.9.1/pkgsrc/p4-build
./configure --prefix=$SDE_INSTALL --with-tofino P4_NAME=basic_switching P4_PATH=/root/bf-sde-8.9.1/pkgsrc/p4-examples/programs/basic_switching/basic_switching.p4 --enable-thrift
make 
make install

#generate .conf 
cd ./bf-sde-8.9.1/pkgsrc/p4-examples
./configure --prefix=$SDE_INSTALL 
make
make install


cd ./bf-sde-8.9.1
 ./run_switchd.sh -p basic_switching
ucli
pm
show
..
..
bf_pltfm
qsfp
qsfp-lpmode-hw 31 0
qsfp-lpmode-hw 32 0
..
../root/bf-sde-8.9.1/pkgsrc/switch-p4-16/p4src/switch-tofino
pm
show
end
pd-basic-switching
pd forward add_entry set_egr ig_intr_md_ingress_port 128 action_egress_spec 136
pd forward add_entry set_egr ig_intr_md_ingress_port 136 action_egress_spec 128


```
pd forward set_default_action ……
```

dump_table forward
end

pm
show


resource:
/root/bf-sde-8.9.1/pkgsrc/p4-build/tofino/tlb/visualization



scp /path/filename username@servername:/path
scp -r  root@192.168.200.2:/root/bf-sde-8.9.1/pkgsrc/p4-build/tofino/lvj/ ./

grep -r register ./*

find ./* -name "intrinsic_metadata.p4"
vi ./install/share/p4c/p4_14include/tofino/intrinsic_metadata.p4 
root@localhost:~/bf-sde-8.9.1/pkgsrc# vi p4-compilers/share/p4_lib/tofino/intrinsic_metadata.p4 


p4i -w $SDE/build &
ip:3000/client

./run_switch.sh -p tlb &
bfshell -f tlbcommands.txt

./run_switchd.sh -p basic_switching &
bfshell -f bs40g.txt
bfshell -f bs40g8path.txt
bfshell -f bs40grpo.txt

配置流表
root@localhost:~/bf-sde-8.9.1# ./run_bfshell.sh -f lvjo40g.txt

./run_switchd.sh -p lvjeasy &
bfshell -f lvjeasy40g.txt
bfshell -f lvj2patheasy40.txt
bfshell -f lvjeasy40g8path.txt

./run_switchd.sh -p lvjorigin &
bfshell -f lvjo40g.txt
bfshell -f lvjo40g8path.txt 


./run_switchd.sh -p lvj &
bfshell -f lvj40g.txt

/root/bf-sde-8.9.1/build/bf-drivers/
make doc
/root/bf-sde-8.9.1/build/bf-drivers/doc

/root/bf-sde-8.9.1/install/include/tofino/pdfixed/pd_tm.h 
/root/bf-sde-8.9.1/pkgsrc/bf-drivers/include/traffic_mgr/traffic_mgr_sch_intf.h
/root/bf-sde-8.9.1/pkgsrc/bf-drivers/include/traffic_mgr/traffic_mgr_q_intf.h
root@localhost:~/bf-sde-8.9.1/build/bf-drivers/pdfixed_thrift

vi ./pkgsrc/switch_test/targets/tofino/switch.conf
/root/bf-sde-8.9.1/install/share/p4/targets/tofino/basic_switching.conf


./install/share/p4c/p4include/core.p4
root@localhost:~/bf-sde-8.9.1# 


./configure --prefix=$SDE_INSTALL --with-tofino P4_NAME=lvjorigin P4_PATH=/root/bf-sde-8.9.1/pkgsrc/p4-examples/programs/lvjorigin/lvjorigin.p4 --enable-thrift

./configure --prefix=$SDE_INSTALL --with-tofino P4_NAME=lvjeasy P4_PATH=/root/bf-sde-8.9.1/pkgsrc/p4-examples/programs/lvjeasy/lvjeasy.p4 --enable-thrift


