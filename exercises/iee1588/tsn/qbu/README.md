cadence has implemented a MAC Merge Sublayer (MMSL) module that is instanced with two Ethernet MAC
 instances, one for the pre-emptable MAC (pMAC) and the other for the express MAC (eMAC).

The eMAC is only required to support a single transmit queue. When pre-emption is disabled, 
the MMSL arbitrates between eMAC and pMAC on a frame-by-frame basis. The eMAC still has highest priority, but the frames transmitted from the pMAC will go out unmodified. For 802.1Qbu, pre-emption requires hardware support and Cadence offer a solution to provide the necessary hardware. Cadence plans to add support for 802.1CB and 802.1Qci 
once the standards become more definite.
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/mac.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/qbu.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/qbu_debug.png)


# mmsl control

echo '0x1,0x0F00,0x28' >  /sys/kernel/eth0/mms_value


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/run.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/tcpdump.png)


# 802.3br
**the ip is The Cadence IP for Gigabit Ethernet MAC with DMA, 1588, and TSN/AVB (IP7014)**

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/3br.png)



![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/pemac.png)

# ping through pmac

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/ping_pmac.png)

# ping through emac

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/ping_emac.png)


# references

[TSN technology: basics of Ethernet Frame Preemption, Part 2](https://iebmedia.com/technology/tsn/tsn-technology-basics-of-ethernet-frame-preemption-part-2/)