#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>

static
void s_send_fd(int socket, int fd)  // send fd by socket
{
	struct msghdr msg = { 0 };
	char buf[CMSG_SPACE(sizeof(fd))];
	memset(buf, '\0', sizeof(buf));


	struct iovec io = { .iov_base = "", .iov_len = 1 };


	msg.msg_iov = &io;
	msg.msg_iovlen = 1;
	msg.msg_control = buf;
	msg.msg_controllen = sizeof(buf);


	struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
	cmsg->cmsg_level = SOL_SOCKET;
	cmsg->cmsg_type = SCM_RIGHTS;
	cmsg->cmsg_len = CMSG_LEN(sizeof(fd));


	memmove(CMSG_DATA(cmsg), &fd, sizeof(fd));


	msg.msg_controllen = CMSG_SPACE(sizeof(fd));


	int err = sendmsg(socket, &msg, 0);
	assert(err >= 0);
}

static
int s_receive_fd(int socket)  // receive fd from socket
{
	struct msghdr msg = {0};


	char m_buffer[1];
	struct iovec io = { .iov_base = m_buffer, .iov_len = sizeof(m_buffer) };
	msg.msg_iov = &io;
	msg.msg_iovlen = 1;


	char c_buffer[256];
	msg.msg_control = c_buffer;
	msg.msg_controllen = sizeof(c_buffer);


	int err = recvmsg(socket, &msg, 0);
	assert(err >= 0);
	
	struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);

	int fd;
	memmove(&fd, CMSG_DATA(cmsg), sizeof(fd));

	return fd;
}

char * curr_time()
{
	time_t mytime = time(NULL);
	char * time_str = ctime(&mytime);
	time_str[strlen(time_str)-1] = '\0';
	return time_str;
}

int main(int argc, char **argv)
{
	assert(argc >= 2);
	const char *filename = argv[1];

	int sv[2];
	int err = socketpair(AF_UNIX, SOCK_DGRAM, 0, sv);
	assert(err == 0);
	printf("Sockets connected.\n");

	sem_t* semptr = (sem_t*)mmap(0, sizeof(sem_t), PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_SHARED, 0, 0 );
	assert((void*)semptr != MAP_FAILED);	

	sem_init(semptr, 1 /*shared*/, 0 /*value*/);
	printf("Semaphore created and initialized with 0.\n\n");

	int pid = fork();
        size_t i = 0;
	if (pid > 0)  // in parent
	{
		printf("PARENT in action.\n");
		close(sv[1]);
		int sock = sv[0];
	
		int fd = open(filename, O_RDWR);
		assert(fd >= 0);
#if 1
                lseek(fd,1024,SEEK_SET);          //文件本身的初始大小
                write(fd,"",1);
#endif
		struct stat statbuf;
		err = fstat(fd, &statbuf);
		assert(err >= 0);

		char *ptr = mmap(NULL, statbuf.st_size,
				PROT_READ | PROT_WRITE,
				MAP_SHARED,
				fd, 0);
		assert(ptr != MAP_FAILED);
                
		printf("PARENT sends file description of %d via Unix domain sockets.\n", fd);
		s_send_fd(sock, fd);

		printf("PARENT reads memory-shared file:\n");
		for (i = 0; i < statbuf.st_size; i++)
			write(1, ptr + i, 1); /* one byte at a time */

		printf("\n");
		printf("PARENT sleeps for 5 seconds... Time: %s\n", curr_time());
		sleep(5);

		printf("PARENT edits memory-shared file. Time: %s\n", curr_time());
		for (i = 0; i < statbuf.st_size / 2; ++i) {
			int j = statbuf.st_size - i - 1;
			int t = ptr[i];
			ptr[i] = ptr[j];
			ptr[j] = t;
		}  

		/* increment the semaphore so that reader can read */
		printf("PARENT increments semaphore. Time: %s\n", curr_time());
		err = sem_post(semptr);
		assert(err >= 0);

		err = munmap(ptr, statbuf.st_size);
		assert(err >= 0);
		close(fd);
		nanosleep(&(struct timespec){ .tv_sec = 1, .tv_nsec = 500000000}, 0);
                munmap(semptr,sizeof(sem_t));
		printf("PARENT exits.\n");
	}
	else  // in child
	{
		printf("CHILD in action.\n");
		close(sv[0]);
		int sock = sv[1];
	
		nanosleep(&(struct timespec){ .tv_sec = 0, .tv_nsec = 500000000}, 0);

		int fd = s_receive_fd(sock);
		printf("CHILD receives %d via Unix domain sockets.\n", fd);
	
		struct stat statbuf;
		err = fstat(fd, &statbuf);
		assert(err >= 0);
	
		char *ptr = mmap(NULL, statbuf.st_size,
				PROT_READ | PROT_WRITE,
				MAP_SHARED,
			        fd, 0);
   		assert(ptr != MAP_FAILED);
				 

		printf("CHILD waits for its parent to increment semaphore. Time: %s\n", curr_time());

		/* use semaphore as a mutex (lock) by waiting for writer to increment it */
		if (!sem_wait(semptr)) { /* wait until semaphore != 0 */					
		       	printf("CHILD can now read. Time: %s\n", curr_time());
			printf("CHILD reads memory-shared file:\n ");
			for (i = 0; i < statbuf.st_size; i++)
				write(1, ptr + i, 1); /* one byte at a time */
			sem_post(semptr);
		}

	    	err = munmap(ptr, statbuf.st_size);
	    	assert(err >= 0);

		printf("\n");
		printf("CHILD exits.\n");
		close(fd);
                munmap(semptr,sizeof(sem_t));
	}
	return 0;
}
