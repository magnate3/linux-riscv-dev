
![images](bbr3.png)
[bbr3-2023](https://research.cec.sc.edu/files/cyberinfra/files/BBR%20-%20Fundamentals%20and%20Updates%202023-08-29.pdf)
[Motivation for BBR evaluation](https://blog.apnic.net/2020/01/10/when-to-use-and-not-use-bbr/)   
```
./configure.sh
./run_tests.sh
./graph_tests.sh
```

```
tests="shallow" ./run_tests.sh
root@ubuntux86:# ls out/bufferbloat/
bbr1:2  bbr:2  cubic:2
root@ubuntux86:# pwd
/work/test/nsperf
root@ubuntux86:# ls out/bufferbloat/
bbr1:2  bbr:2  cubic:2
root@ubuntux86:# ls out/
bufferbloat  coexist  ecn_bulk  random_loss  shallow
root@ubuntux86:# 

```

#  qdisc

```
setup_htb_and_qdisc(d)
```

# graph  

```
tests="shallow" ./graph_tests.sh 
```

def median函数bug修复   
```

def median(nums):
    """Return median of all numbers."""

    if len(nums) == 0:
        return 0
    sorted_nums = sorted(nums)
    n = len(sorted_nums)
    m = n - 1
    return (sorted_nums[int(n/2)] + sorted_nums[int(m/2)]) / 2.0
```


```
root@ubuntux86:# ls graphs/
bufferbloat_50M_30ms_varybuf.png
```