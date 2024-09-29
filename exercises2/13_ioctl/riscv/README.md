

#  Could not register device number

更改#define MY_MAJOR_NUMBER 90 为 #define MY_MAJOR_NUMBER 99  
```
[22015.626947] Hello, Linux kernel!
[22015.626970] Could not register device number
```


```
 cat /proc/devices 
Character devices:
  1 mem
  2 pty
  3 ttyp
  4 /dev/vc/0
  4 tty
  4 ttyS
  5 /dev/tty
  5 /dev/console
  5 /dev/ptmx
  7 vcs
 10 misc
 13 input
 14 sound
 29 fb
 89 i2c
 90 mtd
116 alsa
128 ptm
136 pts
153 spi
180 usb
188 ttyUSB
189 usb_device
226 drm
244 EtherCAT
245 rpmb
246 ttyDBC
247 usbmon
248 nvme-generic
249 nvme
250 ttySIF
251 switchtec
252 bsg
253 rtc
254 gpiochip

Block devices:
  7 loop
  8 sd
 11 sr
 31 mtdblock
 43 nbd
 65 sd
 66 sd
 67 sd
 68 sd
 69 sd
 70 sd
 71 sd
128 sd
129 sd
130 sd
131 sd
132 sd
133 sd
134 sd
135 sd
179 mmc
254 virtblk
259 blkext
```