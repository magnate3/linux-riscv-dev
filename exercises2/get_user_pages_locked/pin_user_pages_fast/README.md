

```
static int dax_lock_page(void *va, struct page **p)
{
	int ret;

	dax_dbg("uva %p", va);

	ret = pin_user_pages_fast((unsigned long)va, 1, FOLL_WRITE, p);
	if (ret == 1) {
		dax_dbg("locked page %p, for VA %p", *p, va);
		return 0;
	}

	dax_dbg("pin_user_pages failed, va=%p, ret=%d", va, ret);
	return -1;
}
```


# centos


```
[root@centos7 pin_user_pages]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 pin_user_pages]# 
```
```
[root@centos7 pin_user_pages]# ./user_test 
data is from kernel is  Mohan
[root@centos7 pin_user_pages]# 
```

```
[704225.207806] sample_open
[704225.210473] sample_write
[704225.213085] kernel < 4.14.4 call pin_user_pages_fast_longterm 
[704225.218980] Got mmaped.

[704225.223169] sample_release
```

# ubuntu


```
root@ubuntux86:# uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntux86:# 
```

```
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 11, 0)
static int pin_user_pages_fast_longterm(unsigned long start, int nr_pages, unsigned int gup_flags, struct page **pages)
{
        pr_info("kernel >= 5.11.0 call pin_user_pages_fast_longterm \n");
        // vma array allocation removed in 52650c8b466bac399aec213c61d74bfe6f7af1a4.
        return pin_user_pages_fast(start, nr_pages, gup_flags | FOLL_LONGTERM, pages);
}
```


```
root@ubuntux86:# insmod  page_test.ko 
root@ubuntux86:# ./user_test 
data is from kernel is  Mohan
root@ubuntux86:# dmesg | tail -n 6
[  385.129549] sample_open
[  385.129635] sample_write
[  385.129638] kernel >= 5.11.0 call pin_user_pages_fast_longterm 
[  385.129643] Got mmaped.

[  385.129677] sample_release
root@ubuntux86:# 
```