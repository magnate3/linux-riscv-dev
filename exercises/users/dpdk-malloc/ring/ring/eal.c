#include <rte_eal.h>
#include "eal_private.h"
#include "eal_memcfg.h"
#include "eal_internal_cfg.h"
#include "eal_filesystem.h"
/* Address of global and public configuration */
static struct rte_mem_config early_mem_config;
/* Address of global and public configuration */
static struct rte_config rte_config = {
                .mem_config = &early_mem_config,
};

/* Return a pointer to the configuration structure */
struct rte_config *
test_rte_eal_get_configuration(void)
{
        return &rte_config;
}
#if 0
struct rte_config {
        uint32_t master_lcore;       /**< Id of the master lcore */
        uint32_t lcore_count;        /**< Number of available logical cores. */
        uint32_t numa_node_count;    /**< Number of detected NUMA nodes. */
        uint32_t numa_nodes[RTE_MAX_NUMA_NODES]; /**< List of detected NUMA nodes. */
        uint32_t service_lcore_count;/**< Number of available service cores. */
        enum rte_lcore_role_t lcore_role[RTE_MAX_LCORE]; /**< State of cores. */

        /** Primary or secondary configuration */
        enum rte_proc_type_t process_type;

        /** PA or VA mapping mode */
        enum rte_iova_mode iova_mode;

        /**
 *          * Pointer to memory configuration, which may be shared across multiple
 *                   * DPDK instances
 *                            */
        struct rte_mem_config *mem_config;
} __attribute__((__packed__));
#endif
struct lcore_config lcore_config[RTE_MAX_LCORE];
/* platform-specific runtime dir */
static char runtime_dir[PATH_MAX];

static const char *default_runtime_dir = "/var/run";
const char *
rte_eal_get_runtime_dir(void)
{
        return runtime_dir;
}
enum rte_proc_type_t
rte_eal_process_type(void)
{
        return rte_config.process_type;
}

extern struct internal_config internal_config; /**< Global EAL configuration. */
int rte_eal_has_hugepages(void)
{
        return ! internal_config.no_hugetlbfs;
}

enum rte_iova_mode
rte_eal_iova_mode(void)
{
        return test_rte_eal_get_configuration()->iova_mode;
}

static int
mark_freeable(const struct rte_memseg_list *msl, const struct rte_memseg *ms,
                void *arg __rte_unused)
{
        /* ms is const, so find this memseg */
        struct rte_memseg *found;

        if (msl->external)
                return 0;

        found = rte_mem_virt2memseg(ms->addr, msl);

        found->flags &= ~RTE_MEMSEG_FLAG_DO_NOT_FREE;

        return 0;
}
int
rte_eal_cleanup(void)
{
        /* if we're in a primary process, we need to mark hugepages as freeable
 *          * so that finalization can release them back to the system.
 *                   */
        if (rte_eal_process_type() == RTE_PROC_PRIMARY)
                rte_memseg_walk(mark_freeable, NULL);
#if HAVE_USE
        rte_service_finalize();
        rte_mp_channel_cleanup();
        eal_cleanup_config(&internal_config);
#endif
        return 0;
}

/* parse a sysfs (or other) file containing one integer value */
int
eal_parse_sysfs_value(const char *filename, unsigned long *val)
{
        FILE *f;
        char buf[BUFSIZ];
        char *end = NULL;

        if ((f = fopen(filename, "r")) == NULL) {
                RTE_LOG(ERR, EAL, "%s(): cannot open sysfs value %s\n",
                        __func__, filename);
                return -1;
        }

        if (fgets(buf, sizeof(buf), f) == NULL) {
                RTE_LOG(ERR, EAL, "%s(): cannot read sysfs value %s\n",
                        __func__, filename);
                fclose(f);
                return -1;
        }
        *val = strtoul(buf, &end, 0);
        if ((buf[0] == '\0') || (end == NULL) || (*end != '\n')) {
                RTE_LOG(ERR, EAL, "%s(): cannot parse sysfs value %s\n",
                                __func__, filename);
                fclose(f);
                return -1;
        }
        fclose(f);
        return 0;
}
const char *
eal_get_hugefile_prefix(void)
{
        if (internal_config.hugefile_prefix != NULL)
                return internal_config.hugefile_prefix;
        return HUGEFILE_PREFIX_DEFAULT;
}
