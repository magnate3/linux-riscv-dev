# Free pages filler
## Brief description

This project was created to save tool that can write and read to free pages of buddy allocator.
It consists of 2 parts:
1. LKM - provides ability to read and write through file interface.
2. usermode tool - read from this interface.

## LKM

LKM registers file interface ``/proc/filler_start``. To write poison into free pages it walks through
all pages of all zones of all nodes and write signature and 'A'*PAGE_SIZE buffer into them.
To read poisoned free page it walks through all pages of all zones of all nodes and try to read
the signature. If success it will dump buffer to userspace.

## Usermode tool

Read and print PAGE_SIZE from ``/proc/filler_start`` interface. 

## Build

1. install build dependencies

Ubuntu && Debian
```sh
sudo apt-get update;
sudo apt-get install build-essentials-$(uname -r)
sudo apt-get install linux-headers-$(uname -r)
```
Centos && RHEL && Fedora
```sh
sudo yum update
sudo yum groupinstall "Development Tools"
sudo yum install kernel-devel
```
2. build lkm
```sh
make clean all
```

3. build tool
```sh
gcc filler_usr.c -o filler_usr
```

## Test
1. load lkm
```sh
insmod free_page_filler.ko
```
2. write poison to free pages
```
echo 0 > /proc/filler_start
```
3. read free page to check.
```sh
./filler_user
```

## Test environment

It was tested on: 
 - ubuntu-xenial with kernel-4.4.0-206-generic.
 - ubuntu-bionic with kernel-4.15.0-142.146.
 - ubuntu-bionic-hwe with kernel-4.18.0-25.26_18.04.1, kernel-kernel-5.0.0-37.40_18.04.1 and kernel-5.3.0-74.70.
 - ubuntu-focal with kernel-5.4.0-72.80.
 - ubuntu-focal-hwe with kernel-5.8.0-53.60_20.04.1.
 - debian8 with linux-3.16.84-1.
 - debian9 with linux-4.9.258-1.
 - debian10 with linux-4.19.181-1.
 - centos6 with kernel-2.6.32-754.el6.
 - centos7 with kernel-3.10.0-1160.25.1.el7.
 - centos8 with kernel-4.18.0-240.22.1.el8_3.
