

[Internet路由之路由表查找算法概述-哈希/LC-Trie树/256-way-mtrie树](https://blog.csdn.net/dog250/article/details/6596046)    

[命题作文：在一棵IPv4地址树中彻底理解IP路由表的各种查找过程](https://www.cnblogs.com/tlnshuju/p/7103471.html)    


# 测试集
[获取广域网 BGP 路由表(Route-Views) & Bgpdump](https://jasper1024.com/jasper/72faxwn23ac7i/)  
[flatritrie Performance benchmarks 测试](https://github.com/blaa/flatritrie)   
[palmtrie](https://github.com/pixos/palmtrie)   
ClassBench-ng: Benchmarking Packet Classification Algorithms in the OpenFlow Era     
+ bgpdump   
+ RouteView和RIPE RIS是部署收集器并使BGP跟踪数据公开可用的两个主要测量项目。    
+ CAIDA的网络流量   


##  lulea_trie_poc


```
#define BENCHMARK_IPS (100000)
void Benchmark(void)
{
  struct       timespec  sooner;
  struct       timespec  later;
  struct       timespec  diff;
  uint32_t    *pu32IPs = NULL;
  unsigned int uIndex  = 0;


  pu32IPs = calloc(BENCHMARK_IPS, sizeof(*pu32IPs));
  if (!pu32IPs)
  {
    printf("Can't allocate benchmark IP list\n");
    exit(1);
  }

  /* Always use the same seed so that benchmark is reproducible. */
  srand(100);
  for (uIndex = 0; uIndex < BENCHMARK_IPS; uIndex++)
  {
    pu32IPs[uIndex] = rand();
  }

  clock_gettime(CLOCK_MONOTONIC, &sooner);
  for (uIndex = 0; uIndex < BENCHMARK_IPS; uIndex++)
  {
    LookupInTree(pu32IPs[uIndex]);
  }
  clock_gettime(CLOCK_MONOTONIC, &later);
  timediff(&sooner, &later, &diff);
  printf("Benchmark: %d Lookups in radix trie took %ld sec %ld nanosec\n", BENCHMARK_IPS, diff.tv_sec, diff.tv_nsec);

  clock_gettime(CLOCK_MONOTONIC, &sooner);
  for (uIndex = 0; uIndex < BENCHMARK_IPS; uIndex++)
  {
    LuleaTrieLookup(pu32IPs[uIndex], pNextHops);
  }
  clock_gettime(CLOCK_MONOTONIC, &later);
  timediff(&sooner, &later, &diff);
  printf("Benchmark: %d Lookups in luleå trie took %ld sec %ld nanosec\n", BENCHMARK_IPS, diff.tv_sec, diff.tv_nsec);

  free(pu32IPs);
}

```