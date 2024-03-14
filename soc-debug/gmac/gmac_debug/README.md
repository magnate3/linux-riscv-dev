

```
module  nicDbg  uevent
# ls /sys/devices/sysfs_macb/nicDbg/
dev1
# ln -s  /sys/devices/sysfs_macb/nicDbg/dev1  eth2
# 
# echo '0x1,0x12,0x6,0x2' > eth2 
[ 5813.592941] gmac_macb_debug: gmac_store read 1, reg add 12 ,page 6, reg value 2
# 
```



```
[ 6423.422740] ------------[ cut here ]------------
[ 6423.427479] RTNL: assertion failed at drivers/net/phy/phylink.c (1291)
[ 6423.434492] WARNING: CPU: 1 PID: 118 at drivers/net/phy/phylink.c:1291 phylink_stop+0xbe/0xc2
[ 6423.443133] Modules linked in: emac(O) gmac_macb_debug(O) pmac(O) [last unloaded: gmac_macb_debug]
[ 6423.452168] CPU: 1 PID: 118 Comm: sh Tainted: G           O      5.14.12-g15c9a49d91df-dirty #192
[ 6423.461064] Hardware name: sifive,hifive-unleashed-a00 (DT)
[ 6423.466654] epc : phylink_stop+0xbe/0xc2
[ 6423.470614]  ra : phylink_stop+0xbe/0xc2
[ 6423.474564] epc : ffffffff80463dd6 ra : ffffffff80463dd6 sp : ffffffd004283c70
[ 6423.481804]  gp : ffffffff81add788 tp : ffffffe0029e1e00 t0 : 0000000000000005
[ 6423.489043]  t1 : 0000000000000064 t2 : ffffffff80000000 s0 : ffffffd004283c90
[ 6423.496282]  s1 : ffffffe001f03600 a0 : 000000000000003a a1 : ffffffff81a84370
[ 6423.503523]  a2 : 0000000000000003 a3 : 0000000000000000 a4 : 593e4ed275c9a500
[ 6423.510763]  a5 : 593e4ed275c9a500 a6 : c0000000ffffefff a7 : ffffffff802d3c16
[ 6423.518002]  s2 : ffffffe004e26000 s3 : ffffffe004e20000 s4 : 0000000000000a80
[ 6423.525239]  s5 : ffffffe004e207c0 s6 : 0000000000000011 s7 : 0000000000000007
[ 6423.532479]  s8 : 00000000000de658 s9 : 00000000000b2000 s10: 00000000000bb468
[ 6423.539719]  s11: 00000000000e1745 t3 : 00000000000f0000 t4 : 0000000000000030
[ 6423.546959]  t5 : ffffffffffffffff t6 : 0000000000000033
[ 6423.552286] status: 0000000200000120 badaddr: 0000000000000000 cause: 0000000000000003
[ 6423.560221] [<ffffffff80463dd6>] phylink_stop+0xbe/0xc2
[ 6423.565496] [<ffffffff01b29c04>] macb_close+0x62/0xc4 [pmac]
[ 6423.575317] [<ffffffff01bb62ea>] gmac_store+0x1a0/0x216 [gmac_macb_debug]
```