#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/nvme_ioctl.h>
#include "linux/nvme.h"
#define NVME_DEVICE_PATH "/dev/nvme1n1"  // NVMe设备路径
#define BUF_LEN 128
int main() {
    int fd = open(NVME_DEVICE_PATH, O_RDWR);  // 打开NVMe设备文件
    if (fd == -1) {
        perror("Failed to open NVMe device");
        return 1;
    }
    void *buf = malloc(BUF_LEN);
    // 设置读写命令参数
    struct nvme_user_io io;
    memset(&io, 0, sizeof(struct nvme_user_io));
    io.opcode = nvme_cmd_write;  // 写操作
    io.addr = (unsigned long)buf;  // 分配缓冲区，假设大小为4KB
    io.slba = 0;  // 起始逻辑块地址（LBA）
    io.nblocks = 1;  // 操作的块数，这里只写入一个块
#if 0
    snprintf(buf, BUF_LEN, "Hello world: regan");
    // 发送命令到NVMe设备
    if (ioctl(fd, NVME_IOCTL_SUBMIT_IO, &io) == -1) {
        perror("Failed to submit IO request");
        close(fd);
        free(buf);
        return 1;
    }
    printf("Write operation completed successfully\n");
    snprintf(buf, BUF_LEN, "good bye: lake");
#endif
    // 修改命令为读操作
    io.opcode = nvme_cmd_read;
    // 发送命令到NVMe设备
     if (ioctl(fd, NVME_IOCTL_SUBMIT_IO, &io) == -1) {
        perror("Failed to submit IO request");
        close(fd);
        free(buf);
        return 1;
    }
    printf("Read operation completed successfully\n");
    printf("read content: %s \n",(char *)(io.addr));
    // 关闭设备文件和释放缓冲区
    close(fd);
    free(buf);
    return 0;
}
