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


# references

[git clone from](https://github.com/eyonggu/tools)
