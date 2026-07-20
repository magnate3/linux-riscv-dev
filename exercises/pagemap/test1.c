#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>

//¼ÆËãĞéÄâµØÖ·¶ÔÓ¦µÄµØÖ·£¬´«ÈëĞéÄâµØÖ·vaddr£¬Í¨¹ıpaddr´«³öÎïÀíµØÖ·
void mem_addr(unsigned long vaddr, unsigned long *paddr)
{
	int pageSize = getpagesize();//µ÷ÓÃ´Ëº¯Êı»ñÈ¡ÏµÍ³Éè¶¨µÄÒ³Ãæ´óĞ¡

	unsigned long v_pageIndex = vaddr / pageSize;//¼ÆËã´ËĞéÄâµØÖ·Ïà¶ÔÓÚ0x0µÄ¾­¹ıµÄÒ³ÃæÊı
	unsigned long v_offset = v_pageIndex * sizeof(uint64_t);//¼ÆËãÔÚ/proc/pid/page_mapÎÄ¼şÖĞµÄÆ«ÒÆÁ¿
	unsigned long page_offset = vaddr % pageSize;//¼ÆËãĞéÄâµØÖ·ÔÚÒ³ÃæÖĞµÄÆ«ÒÆÁ¿
	uint64_t item = 0;//´æ´¢¶ÔÓ¦ÏîµÄÖµ

	int fd = open("/proc/self/pagemap", O_RDONLY);//£¡£ÒÔÖ»¶Á·½Ê½´ò¿ª/proc/pid/page_map
	if(fd == -1)//ÅĞ¶ÏÊÇ·ñ´ò¿ªÊ§°Ü
	{
		printf("open /proc/self/pagemap error\n");
		return;
	}

	if(lseek(fd, v_offset, SEEK_SET) == -1)//½«ÓÎ±êÒÆ¶¯µ½ÏàÓ¦Î»ÖÃ£¬¼´¶ÔÓ¦ÏîµÄÆğÊ¼µØÖ·ÇÒÅĞ¶ÏÊÇ·ñÒÆ¶¯Ê§°Ü
	{
		printf("sleek error\n");
		return;	
	}

	if(read(fd, &item, sizeof(uint64_t)) != sizeof(uint64_t))//¶ÁÈ¡¶ÔÓ¦ÏîµÄÖµ£¬²¢´æÈëitemÖĞ£¬ÇÒÅĞ¶Ï¶ÁÈ¡Êı¾İÎ»ÊıÊÇ·ñÕıÈ·
	{
		printf("read item error\n");
		return;
	}

	if((((uint64_t)1 << 63) & item) == 0)//ÅĞ¶ÏpresentÊÇ·ñÎª0
	{
		printf("page present is 0\n");
		return ;
	}

	uint64_t phy_pageIndex = (((uint64_t)1 << 55) - 1) & item;//¼ÆËãÎïÀíÒ³ºÅ£¬¼´È¡itemµÄbit0-54

	*paddr = (phy_pageIndex * pageSize) + page_offset;//ÔÙ¼ÓÉÏÒ³ÄÚÆ«ÒÆÁ¿¾ÍµÃµ½ÁËÎïÀíµØÖ·
}

const int a = 100;//È«¾Ö³£Á¿

int main()
{
	int b = 100;//¾Ö²¿±äÁ¿
	static c = 100;//¾Ö²¿¾²Ì¬±äÁ¿
	const int d = 100;//¾Ö²¿³£Á¿
	char *str = "Hello World!";

	unsigned long phy = 0;//ÎïÀíµØÖ·

	char *p = (char*)malloc(100);//¶¯Ì¬ÄÚ´æ
	
	int pid = fork();//´´½¨×Ó½ø³Ì
	if(pid == 0)
	{
		//p[0] = '1';//×Ó½ø³ÌÖĞĞŞ¸Ä¶¯Ì¬ÄÚ´æ
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
