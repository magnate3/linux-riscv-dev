#include <stdio.h>
#include <ngx_shmtx.h>
#include <ngx_shmem.h>
#include <config.h>
#include <sys/wait.h>
#include <ngx_slab.h>


int test_metux(void);
int test_slab(void);
int print_info(void);

ngx_shm_t shm = {NULL, 1*1024*1024, NULL, 0};
ngx_shmtx_sh_t *atomic;

ngx_shmtx_t *mtx;
int *num;
int proc_num = 1;
int incr_num = 1000;

int test_metux(void){
    ngx_int_t res;

    shm.size = sizeof(ngx_shmtx_t) + sizeof(ngx_shmtx_sh_t) + 20;
    res = ngx_shm_alloc(&shm);
    if(res != NGX_OK){
        printf("初始化共享内存失败!\n");
        return 0;
    }
    mtx = (ngx_shmtx_t *)shm.addr;
    atomic = (ngx_shmtx_sh_t *)(shm.addr + sizeof(ngx_shmtx_t));

    res = ngx_shmtx_create(mtx, atomic, NULL);

    if(res != NGX_OK){
        printf("初始化锁失败!\n");
        return 0;
    }
    num = (int *)(shm.addr + sizeof(ngx_shmtx_t) + sizeof(ngx_shmtx_sh_t));

    print_info();
}

int test_slab(void){
    ngx_int_t res;
    ngx_slab_sizes_init();
    res = ngx_init_zone_pool(&shm);
    if(res != NGX_OK){
        printf("初始化内存失败!\n");
    }

    ngx_slab_pool_t *pool;
    pool = (ngx_slab_pool_t *) shm.addr;
    num = ngx_slab_alloc(pool, sizeof(int));
    mtx = ngx_slab_alloc(pool, sizeof(ngx_shmtx_t));
    atomic = ngx_slab_alloc(pool, sizeof(ngx_shmtx_sh_t));

    res = ngx_shmtx_create(mtx, atomic, NULL);
    if(res != NGX_OK){
        printf("初始化锁失败!\n");
    }

    return print_info();
}


int print_info(void){
    int i = 0;
    pid_t pid;
    int tmp = 0;

    for(; i < proc_num; ++i){
        pid = fork();
        if(pid < 0){
            printf("进程创建失败!\n");
        }else if(pid == 0){
            init_pid();
            for(i = 0; i < incr_num; ++i){
                printf("进程{%d},第{%d}次操作自增前的值: %d\n",ngx_pid, i, *num);
                ngx_shmtx_lock(mtx);
                tmp = *num;
                *num = tmp + 1;
                ngx_shmtx_unlock(mtx);
                printf("进程{%d},第{%d}次操作自增后的值: %d\n",ngx_pid, i, *num);
            }

            return 0;
        }
    }

    //父进程等待子进程
    for(i = 0; i < proc_num; ++i){
        int status;
        while (1){
            pid = wait(&status);
            if(pid > 0){
                break;
            }
        }
    }

    int total_num = proc_num*incr_num;
    printf("正确结果:%d, 实际结果:%d, 差异:%d\n", total_num, *num, total_num - *num);

    return 0;
}


int main(void){
    init_pid();
    int res;
//    char *name = "test_metux";
    char *name = "test_slab";

//    res = test_metux();
    res = test_slab();
    printf("--------- %s --------\n",name);

    return res;
}
