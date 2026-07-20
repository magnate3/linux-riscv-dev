#include "nodes.h"

static int count_nodes(char **dir_names)
{
    int count = 0;
    int i = 0;
    for (i = 0; dir_names[i] != NULL; ++i)
        if (strncmp(dir_names[i], "node", 4) == 0)
            count += 1;
    return count; 
}

static void get_cpu_list(char cpulist[], char sysNodeBase[100])
{
    FILE *fptr;
    char sysNodeCpuList[100];
    strcpy(sysNodeCpuList, sysNodeBase);
    strcat(sysNodeCpuList, "/cpulist");

    fptr = fopen(sysNodeCpuList, "r");
    size_t n = fread(cpulist, sizeof(char), 100, fptr);
    fclose(fptr);

    cpulist[n] = '\0';
    int len_cpu_list = strlen(cpulist);
    int i;
    for (i = 0; i < len_cpu_list; ++i)
        if (cpulist[i] == '\n')
            cpulist[i] = '\0';
}

static void get_memory(char mem_info_str[], char sysNodeBase[100])
{
    FILE *fptr;
    char memory[256];
    char sysNodeMemory[100];
    int find_digit = 0;

    strcpy(sysNodeMemory, sysNodeBase);
    strcat(sysNodeMemory, "/meminfo");

    fptr = fopen(sysNodeMemory, "r");
    fgets(memory, 256, fptr);
    fclose(fptr);
    int k = 0;
    char unit[8];
    unsigned long mem_kb;

    sscanf(memory, "%*s %*d %*s %lu %s", &mem_kb, unit);
    snprintf(mem_info_str, 100, "%lu %s", mem_kb, unit);
}

static void nodes_data(int nb_node)
{
    char node_subpath[100];
    char sysNodeBase[100];
    char cpulist[100];
    char mem_info_str[256];

    sprintf(node_subpath, "node%d", nb_node);
    strcpy(sysNodeBase, "/sys/devices/system/node/");
    strcat(sysNodeBase, node_subpath);

    get_cpu_list(cpulist, sysNodeBase);
    get_memory(mem_info_str, sysNodeBase);

    printf("Node %d, CPUs %s, %s\n", nb_node, cpulist, mem_info_str);
}

int nodes(char **dir_names)
{
    int nbr_nodes = count_nodes(dir_names);
    int i;

    printf("Detected %d NUMA nodes:\n", nbr_nodes);
    for (i = 0; i < nbr_nodes; ++i)
        nodes_data(i);
    return 0;
}
