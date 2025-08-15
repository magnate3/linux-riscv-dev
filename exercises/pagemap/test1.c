#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>

//���������ַ��Ӧ�ĵ�ַ�����������ַvaddr��ͨ��paddr���������ַ
void mem_addr(unsigned long vaddr, unsigned long *paddr)
{
	int pageSize = getpagesize();//���ô˺�����ȡϵͳ�趨��ҳ���С

	unsigned long v_pageIndex = vaddr / pageSize;//����������ַ�����0x0�ľ�����ҳ����
	unsigned long v_offset = v_pageIndex * sizeof(uint64_t);//������/proc/pid/page_map�ļ��е�ƫ����
	unsigned long page_offset = vaddr % pageSize;//���������ַ��ҳ���е�ƫ����
	uint64_t item = 0;//�洢��Ӧ���ֵ

	int fd = open("/proc/self/pagemap", O_RDONLY);//�����ֻ����ʽ��/proc/pid/page_map
	if(fd == -1)//�ж��Ƿ��ʧ��
	{
		printf("open /proc/self/pagemap error\n");
		return;
	}

	if(lseek(fd, v_offset, SEEK_SET) == -1)//���α��ƶ�����Ӧλ�ã�����Ӧ�����ʼ��ַ���ж��Ƿ��ƶ�ʧ��
	{
		printf("sleek error\n");
		return;	
	}

	if(read(fd, &item, sizeof(uint64_t)) != sizeof(uint64_t))//��ȡ��Ӧ���ֵ��������item�У����ж϶�ȡ����λ���Ƿ���ȷ
	{
		printf("read item error\n");
		return;
	}

	if((((uint64_t)1 << 63) & item) == 0)//�ж�present�Ƿ�Ϊ0
	{
		printf("page present is 0\n");
		return ;
	}

	uint64_t phy_pageIndex = (((uint64_t)1 << 55) - 1) & item;//��������ҳ�ţ���ȡitem��bit0-54

	*paddr = (phy_pageIndex * pageSize) + page_offset;//�ټ���ҳ��ƫ�����͵õ��������ַ
}

const int a = 100;//ȫ�ֳ���

int main()
{
	int b = 100;//�ֲ�����
	static c = 100;//�ֲ���̬����
	const int d = 100;//�ֲ�����
	char *str = "Hello World!";

	unsigned long phy = 0;//�����ַ

	char *p = (char*)malloc(100);//��̬�ڴ�
	
	int pid = fork();//�����ӽ���
	if(pid == 0)
	{
		//p[0] = '1';//�ӽ������޸Ķ�̬�ڴ�
		mem_addr((unsigned long)&a, &phy);
		printf("pid = %d, virtual addr = %x , physical addr = %x\n", getpid(), &a, phy);
	}
	else
	{ 
		mem_addr((unsigned long)&a, &phy);
		printf("pid = %d, virtual addr = %x , physical addr = %x\n", getpid(), &a, phy);
	}

	sleep(100);
	free(p);
	waitpid();
	return 0;
}
