# insmod  mem_map_test.ko 
```
[root@centos7 mem_map]# insmod  mem_map_test.ko 
[root@centos7 mem_map]# dmesg | tail -n 50
[77747.151459] Zone Movable - 0
[77747.154326]   0  0 0 0
[77747.156675] You have 4 node(s) in your system!
[77893.590011] Goodbye, this is exit_mem_map().
[78015.420402] 

********************************************
[78015.428639] Hello, this is init_mem_map().
[78015.432716] You have 8388541 pages to play with!
[78015.437319] node 0 info ****************************************
[78015.443302] node_data[0]->node_start_pfn = 0.
[78015.447637] node_data[0]->node_present_pages = 2097085.
[78015.452838] node_data[0]->node_spanned_pages = 4194304.
[78015.458046] Zone DMA - 1
[78015.460568]   0  18655 65536 32701
[78015.463961] Zone Normal - 1
[78015.466744]   10000  2062237 4128768 2064384
[78015.470996] Zone Movable - 0
[78015.473863]   0  0 0 0
[78015.476217] node 1 info ****************************************
[78015.482195] node_data[1]->node_start_pfn = 4194304.
[78015.487054] node_data[1]->node_present_pages = 2097152.
[78015.492255] node_data[1]->node_spanned_pages = 2097152.
[78015.497462] Zone DMA - 0
[78015.499984]   0  0 0 0
[78015.502333] Zone Normal - 1
[78015.505115]   400000  2095007 2097152 2097152
[78015.509456] Zone Movable - 0
[78015.512323]   0  0 0 0
[78015.514678] node 2 info ****************************************
[78015.520656] node_data[2]->node_start_pfn = 538968064.
[78015.525687] node_data[2]->node_present_pages = 2097152.
[78015.530889] node_data[2]->node_spanned_pages = 2097152.
[78015.536096] Zone DMA - 0
[78015.538618]   0  0 0 0
[78015.540967] Zone Normal - 1
[78015.543749]   20200000  2095005 2097152 2097152
[78015.548263] Zone Movable - 0
[78015.551130]   0  0 0 0
[78015.553484] node 3 info ****************************************
[78015.559462] node_data[3]->node_start_pfn = 541065216.
[78015.564493] node_data[3]->node_present_pages = 2097152.
[78015.569694] node_data[3]->node_spanned_pages = 2097152.
[78015.574901] Zone DMA - 0
[78015.577422]   0  0 0 0
[78015.579771] Zone Normal - 1
[78015.582551]   20400000  2094961 2097152 2097152
[78015.587065] Zone Movable - 0
[78015.589932]   0  0 0 0
[78015.592281] You have 4 node(s) in your system!
```