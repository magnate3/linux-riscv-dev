#/usr/bin/tcsh
sudo umount /mnt/intel
sudo rmmod nvme 
sudo insmod host/nvme.ko
sudo rmmod host/donard_nv_pinbuf.ko
sudo insmod host/donard_nv_pinbuf.ko
sleep 1
sudo chmod 777 /dev/donard_pinbuf
sudo mount /dev/nvme0n1 /mnt/intel
sudo chmod 777 /dev/nvme0n1 


