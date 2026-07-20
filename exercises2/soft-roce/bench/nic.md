
```
/sys/class/infiniband/mlx5_1/ports/1/hw_counters
/sys/class/infiniband/mlx5_1/ports/1/counters
```

```
root@ubuntu:~# ibstatus
Infiniband device 'mlx5_0' port 1 status:
        default gid:     fe80:0000:0000:0000:c670:bdff:feaa:1fc8
        base lid:        0x0
        sm lid:          0x0
        state:           1: DOWN
        phys state:      3: Disabled
        rate:            40 Gb/sec (4X QDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_1' port 1 status:
        default gid:     fe80:0000:0000:0000:c670:bdff:feaa:1fc9
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            100 Gb/sec (4X EDR)
        link_layer:      Ethernet

root@ubuntu:~# mlxlink -d mlx5_1

Operational Info
----------------
State                           : Active
Physical state                  : ETH_AN_FSM_ENABLE
Speed                           : 100G
Width                           : 4x
FEC                             : Standard RS-FEC - RS(528,514)
Loopback Mode                   : No Loopback
Auto Negotiation                : ON

Supported Info
--------------
Enabled Link Speed (Ext.)       : 0x000007f2 (100G_2X,100G_4X,50G_1X,50G_2X,40G,25G,10G,1G)
Supported Cable Speed (Ext.)    : 0x000002f2 (100G_4X,50G_2X,40G,25G,10G,1G)

Troubleshooting Info
--------------------
Status Opcode                   : 0
Group Opcode                    : N/A
Recommendation                  : No issue was observed

Tool Information
----------------
Firmware Version                : 22.36.1010
amBER Version                   : 2.08
MFT Version                     : mft 4.22.1-520

root@ubuntu:~# 
```