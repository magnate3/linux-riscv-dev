#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <err.h>

#define HIBOMA_GET_VERSION 0
#define HIBOMA_OPEN_FD     1

int main()
{
	int r;
	int fd = open("/dev/hiboma", O_RDONLY|O_WRONLY);
        int fd1, fd2;
        char buf[64];
	if ( fd < 0) {
		perror("failed to open /dev/hiboma");
		return 1;
	}

	r = ioctl(fd, HIBOMA_GET_VERSION);
	if (r < 0 ) {
		perror("ioctl");
		return EXIT_FAILURE;
	}
	printf("version = %d\n", r);

	fd1  = ioctl(fd, HIBOMA_OPEN_FD);
	if (fd1 < 0 ) {
		perror("ioctl");
		return EXIT_FAILURE;
	}
	printf("fd = %d\n", fd1);
        read(fd1, buf,8);

	fd2  = ioctl(fd, HIBOMA_OPEN_FD);
	if (fd2  < 0 ) {
		perror("ioctl");
		return EXIT_FAILURE;
	}
	printf("fd = %d\n", fd2);
        read(fd2, buf,8);
	char command[32];
        snprintf(command, sizeof(command), "ls -hal /proc/%d/fd", getpid());
	system(command);
        close(fd2);
        close(fd1);
        close(fd);
        getchar();
	exit(EXIT_SUCCESS);
}
