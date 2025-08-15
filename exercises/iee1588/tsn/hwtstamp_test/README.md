
# client

```
root@Ubuntu-riscv64:~# ./stamp_send  ecat0  192.168.5.82
source IP: 169.254.15.88
Test started.
Sent packet number (0/10) : hello world 0
Hardwave send timestamp: 0 s, 274274035608 ns

Sent packet number (1/10) : hello world 1
Hardwave send timestamp: 0 s, 274274035608 ns

Sent packet number (2/10) : hello world 2
Hardwave send timestamp: 0 s, 274274035608 ns

Sent packet number (3/10) : hello world 3
Hardwave send timestamp: 0 s, 274274035608 ns
```

# server

```
root@ubuntux86:/work/test/tsn/hwtstamp_test# ./stamp_recv  enp0s31f6
source IP: 192.168.5.82
Test started.
Recv pakage: hello world 0
HW: 1678262617 s, 771293651 ns
ts[1]: 0 s, 0 ns
SW: 0 s, 0 ns
Hardwave recv timestamp: 1678262617 s, 771293651 ns

Recv pakage: hello world 1
HW: 1678262618 s, 771604026 ns
ts[1]: 0 s, 0 ns
SW: 0 s, 0 ns
Hardwave recv timestamp: 1678262618 s, 771604026 ns

Recv pakage: hello world 2
HW: 1678262619 s, 771899401 ns
ts[1]: 0 s, 0 ns
SW: 0 s, 0 ns
Hardwave recv timestamp: 1678262619 s, 771899401 ns

Recv pakage: hello world 3
HW: 1678262620 s, 772146151 ns
ts[1]: 0 s, 0 ns
SW: 0 s, 0 ns
Hardwave recv timestamp: 1678262620 s, 772146151 ns

```