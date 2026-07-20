#  HAVE_XRC

```
grep HAVE_XRC -rn  config.h
86:#define HAVE_XRCD 1
```


```
numactl -C 24,26,27,28,30,32,34,36  ./ib_send_bw -d mlx5_1  -x 3 -c XRC  --report_gbits -s 4096 -m 4096     -a  -F   --run_infinitely

************************************
* Waiting for client to connect... *
************************************
```