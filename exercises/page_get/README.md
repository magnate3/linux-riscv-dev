
# insmod  page_test.ko 

```
[86937.080381] page_cache_get_init
[86937.083514] befor page_cache_get pages->_count :8589934591, 1
[86937.089243] after page_cache_get pages->_count :12884901887, 2
[86937.095051] after page_cache_release pages->_count :8589934591, 1
[root@centos7 page_get]# 
```