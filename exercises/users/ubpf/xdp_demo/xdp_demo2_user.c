#include <linux/bpf.h>
#include <linux/if_link.h>
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/resource.h>

#include <bpf/bpf.h>
#include "bpf_load.h"
#include "bpf_util.h"
#include "libbpf.h"
#define IPPROTO_ICMP 1
#define IPPROTO_TCP 6


static int ifindex;
static __u32 xdp_flags;

//便于用户态使用ctrl+c终止xdp程序，并将xdp程序从HOOK点卸载
static void int_exit(int sig)
{    
    //从HOOK点卸载xdp程序
 bpf_set_link_xdp_fd(ifindex, -1, xdp_flags);
 exit(0);
}

//读取xdp针对策略的丢包个数
static void poll_stats(int interval)
{
 //获取cpu逻辑核心数
 unsigned int nr_cpus = bpf_num_possible_cpus();
 const unsigned int nr_keys = 256;
 __u64 values[nr_cpus];
 __u32 key;
 int i;


 while (1) {
  sleep(interval);
         //循环每个映射的索引
  for (key = 0; key < nr_keys; key++) {
   __u64 sum = 0;
            //查询索引对应的值：value
   assert(bpf_map_lookup_elem(map_fd[0], &key, values) == 0);
   for (i = 0; i < nr_cpus; i++)
    //计算每个逻辑cpu上处理的协议统计
    sum += values[i];
   if (sum){
    if(key==6)
    printf("TCP: %10llu pkt\n", sum);
    else if( key == 1)
    printf("ICMP :%10llu pkt\n", sum);

   }
   
  }
 }
}
//用法提示
static void usage(const char *prog)
{
 fprintf(stderr,
  "usage: %s [OPTS] IFINDEX\n\n"
  "OPTS:\n"
  "    -S    use skb-mode\n"
  "    -N    enforce native mode\n",
  prog);
}

int main(int argc, char **argv)
{
 struct rlimit r = {RLIM_INFINITY, RLIM_INFINITY};
 const char *optstr = "SN";
 char filename[256];
 int opt;

 while ((opt = getopt(argc, argv, optstr)) != -1) {
  switch (opt) {
  case 'S':
   xdp_flags |= XDP_FLAGS_SKB_MODE;
   break;
  case 'N':
   xdp_flags |= XDP_FLAGS_DRV_MODE;
   break;
  default:
   usage(basename(argv[0]));
   return 1;
  }
 }

 if (optind == argc) {
  usage(basename(argv[0]));
  return 1;
 }

 if (setrlimit(RLIMIT_MEMLOCK, &r)) {
  perror("setrlimit(RLIMIT_MEMLOCK)");
  return 1;
 }
    //获取运行指定的网卡参数(网卡索引，也就是XDP要HOOK的网卡)
 ifindex = strtoul(argv[optind], NULL, 0);

 snprintf(filename, sizeof(filename), "%s_kern.o", argv[0]);
    //调用load_bpf_file函数，继而调用bpf系统调用将编辑的xdp程序进行加载
 if (load_bpf_file(filename)) {
  printf("%s", bpf_log_buf);
  return 1;
 }

 if (!prog_fd[0]) {
  printf("load_bpf_file: %s\n", strerror(errno));
  return 1;
 }
   
 signal(SIGINT, int_exit);
 signal(SIGTERM, int_exit);
     //使用set)link_xdp_fd函数将XDP程序attach
 if (bpf_set_link_xdp_fd(ifindex, prog_fd[0], xdp_flags) < 0) {
  printf("link set xdp fd failed\n");
  return 1;
 }
 printf("yes\n");

 poll_stats(2);
 return 0;
}
