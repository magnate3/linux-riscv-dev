 # insmod hello_tasklet.ko 
  ```
  [7357298.478127] starting init my_tasklet!
   [7357298.481956] hellow my_tasklet! data:19900722
  ```
 # rmmod hello_tasklet.ko
 
 ```
[root@centos7 tasklet]# dmesg | tail -n 10
[6488017.487730] Ebtables v2.0 unregistered
[6496833.878203] Installing knfsd (copyright (C) 1996 okir@monad.swb.de).
[6496833.945273] NFSD: starting 90-second grace period (net ffff000008f05a00)
[6652375.197412] NFSD: client 10.10.16.82 testing state ID with incorrect client ID
[6652375.205027] NFSD: client 10.10.16.82 testing state ID with incorrect client ID
[7075566.903881] NFSD: client 10.10.16.82 testing state ID with incorrect client ID
[7075566.911548] NFSD: client 10.10.16.82 testing state ID with incorrect client ID
[7357298.478127] starting init my_tasklet!
[7357298.481956] hellow my_tasklet! data:19900722
[7357427.769169] exiting my_tasklet!
 ```
  
 
   
   
   
  