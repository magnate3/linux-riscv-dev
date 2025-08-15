    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/mman.h>
    #include <unistd.h>
    
    
    #define MAP_LEN (10 * 4096 * 4096)
    
    int main(int argc, char **argv)
    {
            char *p;
            int i;
    
    
            puts("before mmap ->please exec: free -m\n");
            getchar();
            p = (char *)mmap(0, MAP_LEN, PROT_READ |PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
            puts("after mmap ->please exec: free -m\n");
            puts("before write....\n");
            getchar();
    
            for(i=0;i <4096 *10; i++)
                    p[4096 * i] = 0x55;
    
    
            puts("after write ->please exec: free -m\n");
    
            getchar();
            munmap(p, MAP_LEN); 
            return 0;
    }                                      
