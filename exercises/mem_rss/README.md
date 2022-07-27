1. run the user process using commad ./test
2. Get the pid of the process from the terminal.
3. make the module.
4. run the module using command: insmod p1_1.ko upid=<pid>
5. Remove the module.
6. Get the output from dmesg.


```
[root@centos7 part1_rss]# insmod p1_1.ko upid=32487

Message from syslogd@centos7 at Jul 26 22:14:33 ...
 kernel:********rss Module start*********
[root@centos7 part1_rss]# dmesg | tail -n 10
[81752.553408] RSS for this virtual area is 4K
[81752.557576] Total pages in the virtual area: 1
[81752.561999] Virtual Address: ffffc9270000 - Physical Address: e8205feb160000 
[81752.569104] Mapped and present pages: 1
[81752.572922] Mapped but not present pages: 2
[81752.577089] Not mapped pages: 0
[81752.580215] RSS for this virtual area is 4K
[81752.584378] Total pages in the virtual area: 3
[81752.588807] Total RSS 100K
[81752.591502] Total Virtual address size 152K
[root@centos7 part1_rss]# dmesg | grep 'rss Module start'  -A 200
[81620.127871] ********rss Module start*********
[81620.132403] mm not present...
[81620.135357] Total RSS 0K
[81620.137890] Total Virtual address size 0K
[81750.141349] ********Exiting Module*********
[81752.151687] ********rss Module start*********
[81752.156210] Virtual Address: 400000 - Physical Address: 20205fe77e0000 
[81752.162793] Mapped and present pages: 1
[81752.166618] Mapped but not present pages: 0
[81752.170782] Not mapped pages: 0
[81752.173907] RSS for this virtual area is 4K
[81752.178078] Total pages in the virtual area: 1
[81752.182500] Virtual Address: 410000 - Physical Address: e0205ff5430000 
[81752.189089] Mapped and present pages: 1
[81752.192907] Mapped but not present pages: 0
[81752.197074] Not mapped pages: 0
[81752.200202] RSS for this virtual area is 4K
[81752.204366] Total pages in the virtual area: 1
[81752.208793] Virtual Address: 420000 - Physical Address: e8205ff7360000 
[81752.215380] Mapped and present pages: 1
[81752.219197] Mapped but not present pages: 0
[81752.223361] Not mapped pages: 0
[81752.226493] RSS for this virtual area is 4K
[81752.230656] Total pages in the virtual area: 1
[81752.235084] Virtual Address: ffff96d30000 - Physical Address: 20203fe55a0000 
[81752.242187] Virtual Address: ffff96d40000 - Physical Address: 20203ff3b50000 
[81752.249297] Virtual Address: ffff96d50000 - Physical Address: 20203fe8860000 
[81752.256405] Virtual Address: ffff96d60000 - Physical Address: 20203fe8d50000 
[81752.263506] Virtual Address: ffff96d70000 - Physical Address: 20203fe16c0000 
[81752.270614] Virtual Address: ffff96d80000 - Physical Address: 20203fe16d0000 
[81752.277722] Virtual Address: ffff96d90000 - Physical Address: 20203ff1100000 
[81752.284824] Virtual Address: ffff96da0000 - Physical Address: 20203fe82c0000 
[81752.291933] Virtual Address: ffff96db0000 - Physical Address: 20203fe7d00000 
[81752.299041] Virtual Address: ffff96dd0000 - Physical Address: 20203ff4650000 
[81752.306149] Virtual Address: ffff96e00000 - Physical Address: 20203fe8940000 
[81752.313250] Virtual Address: ffff96e40000 - Physical Address: 20203fe0020000 
[81752.320357] Virtual Address: ffff96e60000 - Physical Address: 20203fe7c90000 
[81752.327465] Mapped and present pages: 13
[81752.331370] Mapped but not present pages: 10
[81752.335626] Not mapped pages: 0
[81752.338752] RSS for this virtual area is 52K
[81752.343003] Total pages in the virtual area: 23
[81752.347518] Virtual Address: ffff96ea0000 - Physical Address: e0205f9d370000 
[81752.354619] Mapped and present pages: 1
[81752.358443] Mapped but not present pages: 0
[81752.362607] Not mapped pages: 0
[81752.365738] RSS for this virtual area is 4K
[81752.369902] Total pages in the virtual area: 1
[81752.374324] Virtual Address: ffff96eb0000 - Physical Address: e8205feaf40000 
[81752.381432] Mapped and present pages: 1
[81752.385255] Mapped but not present pages: 0
[81752.389418] Not mapped pages: 0
[81752.392544] RSS for this virtual area is 4K
[81752.396712] Total pages in the virtual area: 1
[81752.401135] Virtual Address: ffff96ec0000 - Physical Address: e8205fe44b0000 
[81752.408244] Mapped and present pages: 1
[81752.412061] Mapped but not present pages: 0
[81752.416229] Not mapped pages: 0
[81752.419355] RSS for this virtual area is 4K
[81752.423519] Total pages in the virtual area: 1
[81752.427946] Mapped and present pages: 0
[81752.431765] Mapped but not present pages: 1
[81752.435934] Not mapped pages: 0
[81752.439060] RSS for this virtual area is 0K
[81752.443223] Total pages in the virtual area: 1
[81752.447652] Virtual Address: ffff96ee0000 - Physical Address: 200000008b0000 
[81752.454752] Mapped and present pages: 1
[81752.458575] Mapped but not present pages: 0
[81752.462739] Not mapped pages: 0
[81752.465871] RSS for this virtual area is 4K
[81752.470035] Total pages in the virtual area: 1
[81752.474458] Virtual Address: ffff96ef0000 - Physical Address: 20203ff22f0000 
[81752.481566] Virtual Address: ffff96f00000 - Physical Address: 20203fe8740000 
[81752.488674] Mapped and present pages: 2
[81752.492491] Mapped but not present pages: 0
[81752.496660] Not mapped pages: 0
[81752.499786] RSS for this virtual area is 8K
[81752.503949] Total pages in the virtual area: 2
[81752.508378] Virtual Address: ffff96f10000 - Physical Address: e0205f80ad0000 
[81752.515486] Mapped and present pages: 1
[81752.519304] Mapped but not present pages: 0
[81752.523467] Not mapped pages: 0
[81752.526601] RSS for this virtual area is 4K
[81752.530765] Total pages in the virtual area: 1
[81752.535193] Virtual Address: ffff96f20000 - Physical Address: e8205fe9710000 
[81752.542295] Mapped and present pages: 1
[81752.546119] Mapped but not present pages: 0
[81752.550283] Not mapped pages: 0
[81752.553408] RSS for this virtual area is 4K
[81752.557576] Total pages in the virtual area: 1
[81752.561999] Virtual Address: ffffc9270000 - Physical Address: e8205feb160000 
[81752.569104] Mapped and present pages: 1
[81752.572922] Mapped but not present pages: 2
[81752.577089] Not mapped pages: 0
[81752.580215] RSS for this virtual area is 4K
[81752.584378] Total pages in the virtual area: 3
[81752.588807] Total RSS 100K
[81752.591502] Total Virtual address size 152K
[root@centos7 part1_rss]# 
```
