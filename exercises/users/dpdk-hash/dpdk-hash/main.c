#include <stdio.h>
#include <arpa/inet.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
//#include <rte_log.h>
#include <rte_hash.h>
#include <rte_jhash.h>
#include "util.h"
#include "nat-conf.h"
#include "hashMap.h"
#define METRICS_MAX_NAME_LEN  255
#define METRICS_HASH_SIZE                0x3FFFF
#define METRICS_LOCK_SIZE                0xF
// hash表，也就是实现增删改查等功能
#define HASH_ENTRI_MAXNUM 1<<12
#define HUSH_KEY_COUNT   1<< 4

// 创建Hash key。 此处采用TUPLE5
struct net_key {
	uint32_t sip;
	uint32_t dip;
	uint16_t sport;
	uint16_t dport;
	char proto;
};

static void print_key (struct net_key *key) {
	printf("sip: %x dip: %x sport: %x dport %x proto:%d \n", 
		key->sip, key->dip, key->sport, key->dport, key->proto);	
}

// DPDK 创建表，队列等都需要自定义名字
static struct rte_hash * create_hash_table(const char *name){
	struct rte_hash_parameters *param = (struct rte_hash_parameters *) malloc(sizeof(struct rte_hash_parameters));
	if (!param) return NULL;

	param->name = name;
	param->entries = HASH_ENTRI_MAXNUM;
	param->key_len = sizeof(struct net_key);
	param->hash_func = rte_jhash; //  hash 函数
	param->hash_func_init_val = 0;
	param->socket_id =rte_socket_id();               // 进行NUMA， 这个是表示每一块内存上的寻址。
	
	struct rte_hash *hash = rte_hash_create(param);
	if (hash == NULL) {
		//RTE_LOG(INFO, Hash, "========================\n");		
	}
	return hash;
}
struct nat_config *g_nat_cfg;
static hashMap *g_metrics_fwd_domains = NULL;
 // domain+clinetIp query metrics
typedef struct metrics_domain_clientIp{
    char  domain_name[METRICS_MAX_NAME_LEN];
    uint32_t src_addr; 
    uint64_t requestCount  ;        
    uint64_t lastQueryTime ; // unix time us
    int64_t firstQueryTime ;// unix time us
   }metrics_domain_clientIp_st;
static int metrics_domain_clientIp_query(hashNode *node, void* input){

     uint64_t *p_time_start  = (uint64_t*)input;

     metrics_domain_clientIp_st *mNode = (metrics_domain_clientIp_st*) node->data;
     mNode->requestCount++;
     mNode->lastQueryTime  = *p_time_start;
     return 1;
}
static int metrics_check_equal(char *key, hashNode *node, __attribute__((unused))void *check){
    if (strcmp(key,node->key)==0){
        return 1;
    }
    return 0;
}
static int test2()
{
    char  key[METRICS_MAX_NAME_LEN]= {0};
    uint64_t timeStart = 999;
    const char *domain = "test";
    uint32_t src_addr = 32;
    g_nat_cfg = (struct nat_config *)malloc(sizeof(struct nat_config));
    config_file_load(g_nat_cfg,"nat.cfg", NULL);
    log_open(g_nat_cfg->comm.log_file);
     log_msg(LOG_INFO, "log file : %s \n", g_nat_cfg->comm.log_file);
#if 1
    sprintf(key,"%s-%d",domain, src_addr);
    g_metrics_fwd_domains= hmap_create(METRICS_HASH_SIZE, METRICS_LOCK_SIZE, elfHashDomain,
    metrics_check_equal, metrics_domain_clientIp_query, NULL, NULL); 
    metrics_domain_clientIp_st * newNode = xalloc_zero(sizeof(metrics_domain_clientIp_st));
    newNode->firstQueryTime = newNode->lastQueryTime = timeStart;
    memcpy(newNode->domain_name,domain,strlen(domain));
    newNode->src_addr = src_addr;
    newNode->requestCount = 1;
    hmap_update(g_metrics_fwd_domains, key, NULL, (void*)newNode);
    if (HASH_NODE_FIND == hmap_lookup(g_metrics_fwd_domains, domain, NULL, (void*)&timeStart)){
        printf("find %s \n", domain); 
    }
    if (HASH_NODE_FIND == hmap_lookup(g_metrics_fwd_domains, key, NULL, (void*)&timeStart)){
        printf("find %s  \n", key); 
    }
    hmap_del(g_metrics_fwd_domains, key, NULL); 
    if (HASH_NODE_FIND != hmap_lookup(g_metrics_fwd_domains, key, NULL, (void*)&timeStart)){
        printf("not find %s  \n", key); 
    }
#endif
    hmap_del_all(g_metrics_fwd_domains);
    free(g_nat_cfg);
    return 0;
}

//DPDK 添加方式有四种
int main(int argc, char *argv[]){

	// DPDK 环境初始化
	rte_eal_init(argc, argv);
	int i = 0;
	uint32_t  net_sip;
	uint32_t  net_dip;
	uint16_t  sport;
	uint16_t  dport;
	inet_pton(AF_INET,"192.168.1.1",&net_sip);
	inet_pton(AF_INET,"192.168.2.1", &net_dip);
	sport = htons(5000);
	dport = htons(6000);

	struct rte_hash *hash  = create_hash_table("cuckoo hash table");
	
	for (i = 0; i < HUSH_KEY_COUNT; i++){
		struct net_key *nk = malloc(sizeof(struct net_key));
		
		nk->sip = net_sip + i;
		nk->dip = net_dip + i;
		nk->sport = sport+ i;
		nk->dport = dport + i;
		nk->proto = i % 2;
		// key 
		// key hash
		// key data+
		// key hash, data
		if(i % 4 == 0) {
			//rte_hash_add_key(hash, nk);
		}else if(i % 4 == 1){ // 第二种添加
			hash_sig_t key2 = rte_hash_hash(hash, nk);
			
			rte_hash_add_key_with_hash(hash, nk, key2);

		}
		else if (i %4 == 2){ // 第三种添加

			uint32_t* tmpdata = (uint32_t *)malloc(sizeof(uint32_t));
			*tmpdata = i;
			
			rte_hash_add_key_data(hash, nk, (void *)tmpdata);

		}else {
			hash_sig_t key4 = rte_hash_hash(hash, nk);
			uint32_t* tmp = (uint32_t *)malloc(sizeof(uint32_t));
			*tmp= i;
			rte_hash_add_key_with_hash_data(hash, nk, key4, (void *)tmp);
		}
		
	}
#if 0
	for (i = 0;i < HUSH_KEY_COUNT;i ++) {
		struct net_key *nk = malloc(sizeof(struct net_key));
		nk->sip = net_sip + i;
		nk->dip = net_dip + i;
		nk->sport = sport+ i;
		nk->dport = dport + i;
		nk->proto = i % 2;

		int idx = rte_hash_lookup(hash, nk);
		printf("hash lookup --> sip: %x, idx: %d\n", nk->sip, idx);
	
		rte_hash_del_key(hash, nk);

		free(nk);
  }	
#endif
	
	struct net_key *key = NULL;
	void *value = NULL;
	uint32_t next =0;
	
	while (rte_hash_iterate( hash, (const void **)&key, &value,&next) >= 0){
		print_key(key);
		if (value != NULL)
			printf("value : %u \n", *(uint32_t*)value);
	}
        test2();
	
	return 0;
}
