#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>     //Definition of uint64_t
#include <sys/eventfd.h>

int efd; //Eventfd file descriptor
uint64_t eftd_ctr;

int retval;     //for select()
fd_set rfds;        //for select()

int s;

int main() {

    //Create eventfd
    efd = eventfd(0,0);
    if (efd == -1){
        printf("\nUnable to create eventfd! Exiting...\n");
        exit(EXIT_FAILURE);
    }

    printf("\nefd=%d pid=%d",efd,getpid());

    //Watch efd
    FD_ZERO(&rfds);
    FD_SET(efd, &rfds);

    printf("\nNow waiting on select()...");
    fflush(stdout);

    retval = select(efd+1, &rfds, NULL, NULL, NULL);

    if (retval == -1){
        printf("\nselect() error. Exiting...");
        exit(EXIT_FAILURE);
    } else if (retval > 0) {
        printf("\nselect() says data is available now. Exiting...");
        printf("\nreturned from select(), now executing read()...");
        s = read(efd, &eftd_ctr, sizeof(uint64_t));
        if (s != sizeof(uint64_t)){
            printf("\neventfd read error. Exiting...");
        } else {
            printf("\nReturned from read(), value read = %ld",eftd_ctr);
        }
    } else if (retval == 0) {
        printf("\nselect() says that no data was available");
    }

    printf("\nClosing eventfd. Exiting...");
    close(efd);
    printf("\n");
    exit(EXIT_SUCCESS);
}

