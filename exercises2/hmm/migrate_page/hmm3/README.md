
# 更改dst page的内容

```
const char * str1 = "migrate hello world";
static void cmp_page(const struct page * src, const struct page *dst)
{
        char *addr1 = NULL, *addr2 = NULL;
        char buf[64] = {0};
        unsigned long page_offset;
        addr1 =  page_address(src);
        addr2 =  page_address(dst);
        pr_info("cmp page begin \n");
        if(g_addr < (unsigned long)addr1 || g_addr > (unsigned long)addr1 + PAGE_SIZE)
        {
            pr_info("g_addr %lu , page start %lu, page end %lu\n", g_addr, (unsigned long)addr1, (unsigned long)addr1 + PAGE_SIZE);
            //return;
        }
        if(0 == memcmp(addr1,addr2,PAGE_SIZE))
        {
             page_offset = g_addr & ~PAGE_MASK;
             addr2 += page_offset;
             memcpy(buf,addr2,32);
             pr_info("src and dts page are equal \n");
             pr_info("buf is %s \n", buf);
             memcpy(addr2,str1,strlen(str1)+1);
        }
        else
        {
             pr_info("src and dts page are not  equal \n");
        }
}
```
+ 1)  执行memcpy(addr2,str1,strlen(str1)+1)，更改dst page的内容

# ./user 

```
root@ubuntux86:# make
make -C /lib/modules/5.13.0-39-generic/build \
M=/work/kernel_learn/hmm modules
make[1]: Entering directory '/usr/src/linux-headers-5.13.0-39-generic'
make[1]: Leaving directory '/usr/src/linux-headers-5.13.0-39-generic'
#gcc mmap_test.c  -o mmap_test
gcc -g user.c  -o user
root@ubuntux86:# insmod  test_hmm.ko 
root@ubuntux86:# ./user 
***** before migrate: 
 Physical address is 5635760128
content is hello world 
**** after migrate: 
 Physical address is 8796093014016
content is migrate hello world 
run over 
root@ubuntux86:#
```
+  1) 执行HMM_DMIRROR_MIGRATE后，物理地址发生变化了（migrate_vma_finalize映射到了新的页）   
+  2) 执行HMM_DMIRROR_MIGRATE后， content变了（映射到了新的页）   
+  3） 一开始src and dts page are equal     
```
[ 1326.492604] HMM test module loaded. This is only for testing HMM.
[ 1331.067365] cmp page begin 
[ 1331.067373] g_addr 139694131109888 , page start 18446625070096506880, page end 18446625070096510976
[ 1331.067385] src and dts page are equal 
[ 1331.067386] buf is hello world 
[ 1331.067413] dmirror migrate return val: 0 
```