


[Why did the gain in the STARTUP phase be 2/ln(2)](https://groups.google.com/g/bbr-dev/c/TlQG1UgEyyY/m/6FKB7Ah3AgAJ)    


```
./startup  3.14 | grep ROUND
```


```
root@ubuntu:~/tcp_bbr/startup-sim# ./startup  3.0 | grep ROUND
ROUND: bw: 0.000x t: 0.100000 round: 1 cwnd:     20 pif:      0 bw: 100.000000 pacing_rate: 300.000000 release: 0.003333
ROUND: bw: 0.000x t: 0.200000 round: 2 cwnd:     21 pif:     19 bw: 100.000000 pacing_rate: 300.000000 release: 0.166667
ROUND: bw: 2.000x t: 0.300000 round: 3 cwnd:     41 pif:     39 bw: 200.000000 pacing_rate: 600.000000 release: 0.290363
ROUND: bw: 2.000x t: 0.400000 round: 4 cwnd:     81 pif:     77 bw: 400.000000 pacing_rate: 1200.000000 release: 0.400008
ROUND: bw: 1.950x t: 0.500007 round: 5 cwnd:    159 pif:    151 bw: 780.000000 pacing_rate: 2340.000000 release: 0.500267
ROUND: bw: 1.949x t: 0.600266 round: 6 cwnd:    311 pif:    309 bw: 1520.000000 pacing_rate: 4560.000000 release: 0.600357
ROUND: bw: 2.039x t: 0.700356 round: 7 cwnd:    621 pif:    619 bw: 3100.000000 pacing_rate: 9300.000000 release: 0.700352
ROUND: bw: 2.000x t: 0.800356 round: 8 cwnd:   1241 pif:   1239 bw: 6200.000000 pacing_rate: 18600.000000 release: 0.800352
^C
root@ubuntu:~/tcp_bbr/startup-sim# ./startup  3 | grep ROUND
ROUND: bw: 0.000x t: 0.100000 round: 1 cwnd:     20 pif:      0 bw: 100.000000 pacing_rate: 300.000000 release: 0.003333
ROUND: bw: 0.000x t: 0.200000 round: 2 cwnd:     21 pif:     19 bw: 100.000000 pacing_rate: 300.000000 release: 0.166667
ROUND: bw: 2.000x t: 0.300000 round: 3 cwnd:     41 pif:     39 bw: 200.000000 pacing_rate: 600.000000 release: 0.290363
ROUND: bw: 2.000x t: 0.400000 round: 4 cwnd:     81 pif:     77 bw: 400.000000 pacing_rate: 1200.000000 release: 0.400008
ROUND: bw: 1.950x t: 0.500007 round: 5 cwnd:    159 pif:    151 bw: 780.000000 pacing_rate: 2340.000000 release: 0.500267
ROUND: bw: 1.949x t: 0.600266 round: 6 cwnd:    311 pif:    309 bw: 1520.000000 pacing_rate: 4560.000000 release: 0.600357
ROUND: bw: 2.039x t: 0.700356 round: 7 cwnd:    621 pif:    619 bw: 3100.000000 pacing_rate: 9300.000000 release: 0.700352
ROUND: bw: 2.000x t: 0.800356 round: 8 cwnd:   1241 pif:   1239 bw: 6200.000000 pacing_rate: 18600.000000 release: 0.800352
ROUND: bw: 2.000x t: 0.900356 round: 9 cwnd:   2481 pif:   2479 bw: 12400.000000 pacing_rate: 37200.000000 release: 0.900352
ROUND: bw: 2.000x t: 1.000356 round: 10 cwnd:   4961 pif:   4959 bw: 24800.000000 pacing_rate: 74400.000000 release: 1.000352
^C
root@ubuntu:~/tcp_bbr/startup-sim# ./startup  2 | grep ROUND
ROUND: bw: 0.000x t: 0.100000 round: 1 cwnd:     20 pif:      0 bw: 100.000000 pacing_rate: 200.000000 release: 0.005000
ROUND: bw: 0.000x t: 0.200000 round: 2 cwnd:     21 pif:     19 bw: 100.000000 pacing_rate: 200.000000 release: 0.200000
ROUND: bw: 2.000x t: 0.300000 round: 3 cwnd:     41 pif:     25 bw: 200.000000 pacing_rate: 400.000000 release: 0.300544
ROUND: bw: 1.300x t: 0.400544 round: 4 cwnd:     67 pif:     42 bw: 260.000000 pacing_rate: 520.000000 release: 0.401500
ROUND: bw: 1.654x t: 0.501500 round: 5 cwnd:    110 pif:     71 bw: 430.000000 pacing_rate: 860.000000 release: 0.502177
ROUND: bw: 1.674x t: 0.602177 round: 6 cwnd:    182 pif:    111 bw: 720.000000 pacing_rate: 1440.000000 release: 0.602626
ROUND: bw: 1.556x t: 0.702626 round: 7 cwnd:    294 pif:    179 bw: 1120.000000 pacing_rate: 2240.000000 release: 0.702707
ROUND: bw: 1.607x t: 0.802707 round: 8 cwnd:    474 pif:    289 bw: 1800.000000 pacing_rate: 3600.000000 release: 0.802811
ROUND: bw: 1.611x t: 0.902810 round: 9 cwnd:    764 pif:    460 bw: 2900.000000 pacing_rate: 5800.000000 release: 0.902855
ROUND: bw: 1.590x t: 1.002855 round: 10 cwnd:   1225 pif:    737 bw: 4610.000000 pacing_rate: 9220.000000 release: 1.002911
ROUND: bw: 1.601x t: 1.102910 round: 11 cwnd:   1963 pif:   1179 bw: 7380.000000 pacing_rate: 14760.000000 release: 1.102928
ROUND: bw: 1.599x t: 1.202927 round: 12 cwnd:   3143 pif:   1882 bw: 11800.000000 pacing_rate: 23600.000000 release: 1.202929
ROUND: bw: 1.596x t: 1.302929 round: 13 cwnd:   5026 pif:   3008 bw: 18830.000000 pacing_rate: 37660.000000 release: 1.302939
ROUND: bw: 1.598x t: 1.402938 round: 14 cwnd:   8035 pif:   4806 bw: 30090.000000 pacing_rate: 60180.000000 release: 1.402949
ROUND: bw: 1.598x t: 1.502948 round: 15 cwnd:  12842 pif:   7675 bw: 48070.000000 pacing_rate: 96140.000000 release: 1.502951
```