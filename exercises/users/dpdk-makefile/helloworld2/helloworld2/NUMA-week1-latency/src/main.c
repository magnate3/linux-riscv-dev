
#include <sys/shm.h>
#include "main.h"

#define HUGE_PAGESIZE (1 << 29)
void free_names(char **dir_names)
{
    int i;
    for (i = 0; dir_names[i] != NULL; ++i)
        free(dir_names[i]);
    free(dir_names);
}
void test_mbind()
{
    int shm_key, shm_id, ret;
    const unsigned long nodemask = (1ul << (unsigned long)0);
    shm_key = 99999999;
    unsigned int size = SIZE;
    //size = (size + HUGE_PAGESIZE) & ~(HUGE_PAGESIZE - 1);
    shm_id = shmget(shm_key, size, IPC_CREAT | IPC_EXCL | 0666 | SHM_HUGETLB);
    char *shm_buf = (char *)shmat(shm_id, NULL, 0);
    shmctl(shm_id, IPC_RMID, NULL); 
    ret = mbind(shm_buf, size, MPOL_BIND, &nodemask, 2, 0);
    if (0 != ret )
       printf("hugeAlloc: mbind() failed. Key %d \n" ,shm_key);
}
int main(int argc, char *argv[])
{
    struct dirent *pDirent;
    DIR *pDir;
    char **dir_names = malloc(sizeof(char *) * 1024);

    test_mbind();
    pDir = opendir("/sys/devices/system/node/");
    if (pDir == NULL) {
        printf("Command open dir");
        return 1;
    }
    int i = 0;
    while ((pDirent = readdir(pDir)) != NULL) {
        dir_names[i] = malloc(sizeof(char) * strlen(pDirent->d_name) + 1);
        // dir_names[i] = pDirent->d_name;
        strcpy(dir_names[i], pDirent->d_name);
        i++;
    }
    closedir(pDir);
    dir_names[i] = NULL;
    nodes(dir_names);
    latency(0, 0);
    latency(1, 0);
    latency(0, 2);
    latency(1, 2);
    free_names(dir_names);
    return 0;
}
