# tools

- **pagemap**: prints process's physical pages for a given virtual address range.

  Usage:  pagemap <pid> <start va> <end va>

- **pcimem**: reads/writes pci device memory by mmap BAR resource.

  Usage:  pcimem [r|w] <bb:dd:ff> <bar> <offset> <width> [<data>]

- **qemu-make-debian-root**: makes a bebian root image

  Usage: qemu-make-debian-root 1024 stable http://deb.debian.org/debian/ debian.img

         qemu-system-x86_64 -enable-kvm -cpu host -nographic \
                -kernel buildroot-2022.02/output/images/bzImage \
                -hda /home/eyonggu/Workspace/debian.img \
                -append "root=/dev/sda1 rw console=ttyS0" \
                -nic user,model=e1000,hostfwd=tcp::5555-:22
                -nic tap,model=e1000,ifname=tap0

- **dpdk-devbind.py**: binds NIC device to Linux drivers

  Usage: dpdk-devbind.py --help

- **list_sriov-vf.sh**: list VF:s of SR-IOV NIC

# pcimem
```
[root@centos7 user]# ./pcimem r 05:00.0 

Usage:  ./pcimem { op } { bfd } { bar } { offset } { width } [ data ]
        op      : operation type: [r]ead, [w]rite 
        bdf     : bdf of device to act on, e.g. 02:00.0 
        bar     : pci bar/memory region to act on
        offset  : offset into pci memory region
        width   : number of bytes
        data    : data to be written

[root@centos7 user]# ./pcimem r 05:00.0 0 0 4 0
[root@centos7 user]# 
```

# references

[git clone from](https://github.com/eyonggu/tools)
